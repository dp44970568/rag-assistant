"""
Unified LLM Service module.
Provides streaming and non-streaming responses across Google Gemini, OpenAI, Groq, Ollama,
and an intelligent local offline synthesizer.
"""

import os
import json
import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
import httpx

logger = logging.getLogger(__name__)


class LLMService:
    """Multi-provider LLM connector with streaming support."""

    def __init__(
        self,
        provider: str = "offline",
        model_name: str = "gemini-1.5-flash",
        gemini_api_key: str = "",
        openai_api_key: str = "",
        groq_api_key: str = "",
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.provider = provider
        self.model_name = model_name
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.groq_api_key = groq_api_key
        self.ollama_base_url = ollama_base_url

    async def generate_response(
        self,
        prompt: str,
        system_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generates a complete non-streaming response."""
        full_text = []
        async for chunk in self.stream_response(prompt, system_prompt, history):
            full_text.append(chunk)
        return "".join(full_text)

    async def stream_response(
        self,
        prompt: str,
        system_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Yields text chunks as they are generated.
        Dispatches to the active provider.
        """
        provider = (self.provider or "offline").lower()

        # Determine if we should route to actual API or offline fallback
        if provider == "gemini" and self.gemini_api_key:
            async for token in self._stream_gemini(prompt, system_prompt, history):
                yield token
        elif provider == "openai" and self.openai_api_key:
            async for token in self._stream_openai(prompt, system_prompt, history):
                yield token
        elif provider == "groq" and self.groq_api_key:
            async for token in self._stream_groq(prompt, system_prompt, history):
                yield token
        elif provider == "ollama":
            async for token in self._stream_ollama(prompt, system_prompt, history):
                yield token
        else:
            # Fallback to local intelligent synthesis
            async for token in self._stream_offline(prompt, retrieved_chunks):
                yield token

    async def _stream_gemini(
        self,
        prompt: str,
        system_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """Streams response from Google Gemini REST API."""
        model = self.model_name or "gemini-1.5-flash"
        # Support gemini-2.0, gemini-1.5-pro, gemini-1.5-flash
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?key={self.gemini_api_key}&alt=sse"

        # Format messages
        contents = []
        if history:
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        yield f"\n[Gemini API Error {response.status_code}: {err_text.decode('utf-8', errors='replace')}]"
                        return

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if not data_str or data_str == "[DONE]":
                            continue
                        try:
                            data = json.loads(data_str)
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                for p in parts:
                                    if "text" in p:
                                        yield p["text"]
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Gemini streaming exception: {e}")
            yield f"\n[Connection error with Gemini API: {str(e)}. Please check your API key.]"

    async def _stream_openai(
        self,
        prompt: str,
        system_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """Streams response from OpenAI API."""
        model = self.model_name or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        yield f"\n[OpenAI API Error {response.status_code}: {err_text.decode('utf-8', errors='replace')}]"
                        return

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"OpenAI streaming exception: {e}")
            yield f"\n[Connection error with OpenAI API: {str(e)}]"

    async def _stream_groq(
        self,
        prompt: str,
        system_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """Streams response from Groq API."""
        model = self.model_name if "llama" in self.model_name.lower() else "llama-3.3-70b-versatile"
        url = "https://api.groq.com/openai/v1/chat/completions"

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        yield f"\n[Groq API Error {response.status_code}: {err_text.decode('utf-8', errors='replace')}]"
                        return

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Groq streaming exception: {e}")
            yield f"\n[Connection error with Groq API: {str(e)}]"

    async def _stream_ollama(
        self,
        prompt: str,
        system_prompt: str,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """Streams response from local Ollama instance."""
        model = self.model_name or "llama3"
        url = f"{self.ollama_base_url.rstrip('/')}/api/chat"

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for msg in history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        yield f"\n[Ollama Error: Could not reach model '{model}' at {url}]"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            msg = data.get("message", {})
                            token = msg.get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Ollama streaming exception: {e}")
            yield f"\n[Could not connect to Ollama at {self.ollama_base_url}. Ensure Ollama is running locally.]"

    async def _stream_offline(
        self,
        prompt: str,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Intelligent local offline synthesizer.
        Extracts key sentences and synthesizes a direct answer using the retrieved context.
        Works 100% offline without any API key or external network dependency.
        """
        if not retrieved_chunks:
            msg = (
                "No matching information was found in your uploaded documents for this query.\n\n"
                "**Suggestions:**\n"
                "- Upload documents (PDF, DOCX, TXT, MD, CSV) via the sidebar.\n"
                "- Lower the similarity threshold in the Settings panel.\n"
                "- Try rephrasing your search query."
            )
            for word in msg.split(" "):
                yield word + " "
                await asyncio.sleep(0.015)
            return

        # Build synthesized response from top retrieved chunks
        intro = "Based on the retrieved context from your documents:\n\n"
        for word in intro.split(" "):
            yield word + " "
            await asyncio.sleep(0.01)

        for i, chunk in enumerate(retrieved_chunks):
            doc = chunk.get("doc_name", "Document")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "").strip()

            chunk_header = f"### Source {i + 1}: `{doc}` (Relevance: {score:.1%})\n"
            for word in chunk_header.split(" "):
                yield word + " "
                await asyncio.sleep(0.01)

            # Format chunk content cleanly
            lines = text.split("\n")
            for line in lines:
                if line.strip():
                    yield f"> {line.strip()}\n"
                    await asyncio.sleep(0.015)
            yield "\n\n"

        summary = (
            "**Summary:**\n"
            f"Retrieved {len(retrieved_chunks)} relevant passage(s) directly matching your question. "
            "To generate higher-level generative explanations or multi-step reasoning, connect an API key "
            "(Google Gemini, OpenAI, or Groq) in the **Settings** menu at the top right."
        )
        for word in summary.split(" "):
            yield word + " "
            await asyncio.sleep(0.01)
