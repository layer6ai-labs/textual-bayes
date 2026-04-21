import logging
from collections import deque
from copy import deepcopy

from textgrad import logger
from textgrad.optimizer.optimizer import TextualGradientDescent
from textgrad.variable import Variable


class TextualGradientDescentLogProb(TextualGradientDescent):
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            # engine holds httpx clients with RLocks that can't be copied; share the reference
            if k == "engine":
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def logprobs(self, parameters_new: list[Variable]):
        parameters_new_logprobs = []
        for parameter, parameter_new in zip(self.parameters, parameters_new):
            prompt_update_parameter = self._update_prompt(parameter)
            new_text = (
                f"{self.new_variable_tags[0]}{parameter_new.value}{self.new_variable_tags[1]}"
            )
            logprobs = self.engine.logprobs(
                prompt_update_parameter,
                system_prompt=self.optimizer_system_prompt,
                response_text=new_text,
            )
            logprobs = self.extract_logprobs_between_tags(logprobs)
            parameters_new_logprobs.append(logprobs)
        return parameters_new_logprobs

    def step(self):
        parameters_new_logprobs = []
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            new_text, logprobs = self.engine(
                prompt_update_parameter, system_prompt=self.optimizer_system_prompt, logprobs=True
            )
            logger.info(
                "TextualGradientDescentLogProb optimizer response",
                extra={"optimizer.response": new_text},
            )
            try:
                new_value = (
                    new_text.split(self.new_variable_tags[0])[1]
                    .split(self.new_variable_tags[1])[0]
                    .strip()
                )
            # Check if we got a cannot be indexed error
            except IndexError:
                logger.error(
                    f"TextualGradientDescentLogProb optimizer response could not be indexed",
                    extra={"optimizer.response": new_text},
                )
                raise IndexError(
                    f"TextualGradientDescentLogProb optimizer response could not be indexed. This can happen if the optimizer model cannot follow the instructions. You can try using a stronger model, or somehow reducing the context of the optimization. Response: {new_text}"
                )
            parameter.set_value(new_value)
            logger.info(
                "TextualGradientDescentLogProb updated text",
                extra={"parameter.value": parameter.value},
            )
            if self.verbose:
                print("-----------------------TextualGradientDescent------------------------")
                print(parameter.value)

            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)

            logprobs = self.extract_logprobs_between_tags(logprobs)
            parameters_new_logprobs.append(logprobs)
        return parameters_new_logprobs

    def extract_logprobs_between_tags(self, logprobs):
        logger = logging.getLogger("OPT")
        buffer = deque()
        start_tag, end_tag = self.new_variable_tags
        max_tag_len = 1 + max(len(start_tag), len(end_tag))
        start_index = None
        end_index = None

        for i in range(len(logprobs.tokens) - 1, -1, -1):
            token = logprobs.tokens[i]
            buffer.appendleft(token)
            buffer_text = "".join(buffer)
            if end_tag in buffer_text:
                end_index = i + 1
            if start_tag in buffer_text:
                offset = 0
                while start_tag not in "".join(logprobs.tokens[i : (i + offset)]):
                    offset += 1
                start_index = i + offset
                break
            if len(buffer_text) > max_tag_len:
                buffer.pop()
        logger.info(
            f"Selected text by extract between tags: {''.join(logprobs.tokens[start_index:end_index])}"
        )
        return [p for p in logprobs.token_logprobs[start_index:end_index]]
