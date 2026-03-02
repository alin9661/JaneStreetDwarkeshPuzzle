"""API wrapper with batching helpers."""

from typing import Optional

from jsinfer import (
    ActivationsRequest,
    BatchInferenceClient,
    ChatCompletionRequest,
    Message,
)

from .config import API_KEY, log


class PuzzleClient:
    """Thin wrapper around BatchInferenceClient with convenience methods."""

    def __init__(self, api_key: str = API_KEY):
        self.client = BatchInferenceClient()
        self.client.set_api_key(api_key)

    async def chat(
        self,
        prompts: list[dict],
        model: str,
        batch_tag: str = "batch",
    ) -> dict:
        """Send chat completion requests.

        Args:
            prompts: list of dicts with keys:
                - 'messages': list of {'role': str, 'content': str}
                - 'id': optional custom_id
            model: model name
            batch_tag: prefix for custom_ids
        Returns:
            dict mapping custom_id -> response text
        """
        requests = []
        for i, p in enumerate(prompts):
            cid = p.get("id", f"{batch_tag}-{i:04d}")
            msgs = [Message(role=m["role"], content=m["content"]) for m in p["messages"]]
            requests.append(ChatCompletionRequest(custom_id=cid, messages=msgs))

        log.info(f"Sending {len(requests)} chat requests to {model}...")
        try:
            results = await self.client.chat_completions(requests, model=model)
        except Exception:
            log.exception(f"chat_completions failed for {len(requests)} requests on {model}")
            raise

        output = {}
        for cid, resp in results.items():
            text = ""
            if resp.messages:
                text = resp.messages[-1].content
            output[cid] = text
        return output

    async def chat_single(
        self,
        content: str,
        model: str,
        system: Optional[str] = None,
        history: Optional[list[dict]] = None,
    ) -> str:
        """Send a single chat message and return the response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": content})
        results = await self.chat(
            [{"messages": messages, "id": "single"}], model=model
        )
        return results.get("single", "")

    async def get_activations(
        self,
        prompts: list[dict],
        module_names: list[str],
        model: str,
        batch_tag: str = "act",
    ) -> dict:
        """Get activations for given prompts and modules.

        Args:
            prompts: list of dicts with 'messages' and optional 'id'
            module_names: list of module path strings
            model: model name
        Returns:
            dict mapping custom_id -> {module_name: np.ndarray}
        """
        requests = []
        for i, p in enumerate(prompts):
            cid = p.get("id", f"{batch_tag}-{i:04d}")
            msgs = [Message(role=m["role"], content=m["content"]) for m in p["messages"]]
            requests.append(
                ActivationsRequest(custom_id=cid, messages=msgs, module_names=module_names)
            )

        log.info(
            f"Requesting activations from {len(requests)} prompts, "
            f"{len(module_names)} modules on {model}..."
        )
        try:
            results = await self.client.activations(requests, model=model)
        except Exception:
            log.exception(f"activations failed for {len(requests)} requests on {model}")
            raise

        output = {}
        for cid, resp in results.items():
            output[cid] = resp.activations
        return output
