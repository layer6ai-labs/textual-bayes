import logging

from textgrad.engine.together import ChatTogether


class ChatTogetherLogProb(ChatTogether):
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        response_text: str = None,
        echo: bool = False,
        logprobs: bool = False,
    ):
        logger = logging.getLogger("TOGETHER")
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        messages = [
            {"role": "system", "content": sys_prompt_arg},
            {"role": "user", "content": prompt},
        ]
        if response_text is not None:
            messages.append({"role": "assistant", "content": response_text})

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=messages,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )

        if logprobs:
            if response_text is None:
                returned_logprobs = response.choices[0].logprobs
                response_text = response.choices[0].message.content
                logger.info(
                    f"Call to LLM for generation with logprob. LLM generation: {response_text}"
                )
            else:
                returned_logprobs = response.prompt[0].logprobs
                response_text = response.prompt[0].text
                # These tokens are LLM-specific
                delim = [
                    "<|eot_id|>",
                    "<|start_header_id|>",
                    "assistant",
                    "<|end_header_id|>",
                    "\n\n",
                ]
                # Together API changed their API where now `delim` is always appended to the end of the response text
                for i in range(len(returned_logprobs.tokens) - len(delim), len(delim) - 1, -1):
                    if "".join(returned_logprobs.tokens[i - len(delim) : i]) == "".join(delim):
                        returned_logprobs.tokens = returned_logprobs.tokens[
                            i : len(returned_logprobs.tokens) - len(delim)
                        ]
                        returned_logprobs.token_logprobs = returned_logprobs.token_logprobs[
                            i : len(returned_logprobs.token_logprobs) - len(delim)
                        ]
                        returned_logprobs.token_ids = returned_logprobs.token_ids[
                            i : len(returned_logprobs.token_ids) - len(delim)
                        ]
                        logger.info(f"Found the last round at {i}")
                        response_text = "".join(returned_logprobs.tokens)
                        break
                else:
                    logger.info(f"Could not find the last round. Using full text.")
                logger.info(f"Call to LLM for logprob of given text. Logprob on: {response_text}")
            return response_text, returned_logprobs
        else:
            logger.info(
                f"Call to LLM for generation without logprob. LLM generation: {response.choices[0].message.content}"
            )
            return response.choices[0].message.content

    def logprobs(self, prompt: str, response_text: str, system_prompt: str = None):
        _, logprobs = self.generate(
            prompt,
            system_prompt=system_prompt,
            max_tokens=1,
            response_text=response_text,
            echo=True,
            logprobs=True,
        )
        return logprobs
