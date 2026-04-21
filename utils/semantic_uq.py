import logging
from typing import List

import numpy as np
import scipy
from scipy.special import logsumexp
from textgrad import get_engine

from utils.entailment_model import EntailmentDeberta

logger = logging.getLogger("SemanticUQ")


class SemanticClustering:
    def __init__(
        self,
        entailment_model="deberta",
        strict_entailment=False,
        llm_engine="gpt-4o",
        cuda=False,
    ):
        self.entailment_model = None
        self.strict_entailment = strict_entailment
        self.engine = get_engine(llm_engine)

    def check_equivalence_by_entailment(self, text1, text2):
        implication_1 = self.entailment_model.check_implication(text1, text2)
        implication_2 = self.entailment_model.check_implication(text2, text1)
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if self.strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    def get_correctness_by_llm(self, ground_truth_answer, generated_answer, question):
        """
        Uses LLM to check if two texts are semantically equivalent.
        """
        prompt = (
            f"Please act as an answer judge and evaluate the response provided by an AI assistant to the user question displayed below given the ground truth answer.\n"
            f"Your evaluation should consider factors such as if the answers are semantically equivalent given the question. Be as objective as possible.\n"
            f"Please answer with 'yes' or 'no'.\n"
            f"[Question]\n"
            f"{question}\n"
            f"[Generated Answer]\n"
            f"{generated_answer}\n"
            f"[Ground Truth Answer]\n"
            f"{ground_truth_answer}\n"
            f"Just return the 'yes' or 'no' answer. Do not provide any additional content."
        )
        response = self.engine(prompt)
        return "yes" in response.strip().lower()

    def check_equivalence_by_exact_match(self, text1, text2):
        pass

    def check_equivalence_by_token_similarity(self, text1, text2):
        pass

    def _get_semantic_set_ids_from_str(self, semantic_set_ids_str: str) -> List[int]:
        """
        Helper function which takes the string describing semantic clusters outputted by an LLM and produces the
        corresponding list of integers, e.g. "[0, 0, 1]" -> [0, 0, 1].
        """
        return [int(x) for x in semantic_set_ids_str.strip("[]").split(", ")]

    def get_clustering_prompt(self, responses: List[str], prompt: str or None = None) -> str:
        """
        Given a prompt and a corresponding list of responses, prepares the prompt used to ask an LLM to semantically
        cluster the responses. The prompt is optional, and not providing it will slightly change the phrasing of the
        constructed prompt.
        There is currently a lot of hard-coding here and the phrasing of the clustering prompt should be passed as
        part of the config in the future.
        """
        if prompt is None:
            clustering_prompt = (
                f"The following {len(responses)} responses to a question were collected:"
            )
        else:
            clustering_prompt = (
                f"Given the question '{prompt}', {len(responses)} responses were collected."
            )
        clustering_prompt += (
            "\n Can you please cluster these responses into semantically equivalent clusters?"
        )
        if prompt is not None:
            clustering_prompt += " Clustering should be within the context of the question."
        clustering_prompt += "\n Here are the responses"
        for i, res in enumerate(responses):
            clustering_prompt += f"\n {i}. {res}"
        clustering_prompt += (
            f" Your answer should be formatted as [cluster_id_of_response0, "
            f"cluster_id_of_response1, ..., cluster_id_of_response{len(responses)-1}] and should "
            f"not include any additional content. Make sure the order of your answer corresponds to"
            f" the inputs. Each cluster_id should be an integer between 0 "
            f"and {len(responses)-1}, the cluster_id of response 0 should be 0, and you should not "
            f"jump ids (for example, [0, 0, 1] is valid, [0, 1, 0] is valid, but [0, 0, 2] is not)."
        )
        logger.info(f"\nClustering prompt: {clustering_prompt}\n")
        return clustering_prompt

    def get_semantic_ids(
        self, responses: List[str], prompt: str or None = None, method: str = "entailment"
    ) -> List[int]:
        """
        Assigns a unique id to each semantically equivalent response.

        responses: List of responses to be compared
        prompt: str with the prompt used to generate the responses, optional

        returns: List of ids, where each id corresponds to a semantically equivalent response.
        """

        # Choose the method to check equivalence between two texts.
        # 1. entailment
        # 2. exact match (only applicable in multiple choice, or yes or no questions)
        # 3. semantic similarity via cosine similarity (use BERT to check the similarity)
        if method == "entailment":
            check_equivalence = self.check_equivalence_by_entailment
        elif method == "exact_match":
            raise NotImplementedError  # TODO
        elif method == "semantic_similarity":
            raise NotImplementedError  # TODO
        elif method == "llm":
            clustering_prompt = self.get_clustering_prompt(responses, prompt)
            semantic_set_ids = self.engine(
                clustering_prompt
            )  # NOTE: maybe update to not use textgrad?
            return self._get_semantic_set_ids_from_str(semantic_set_ids)

        # Initialize all ids with -1.
        semantic_set_ids = [-1] * len(responses)
        # Keep track of current id.
        semantic_id = 0
        for i, response in enumerate(responses):
            # Check if string1 already has an id assigned.
            if semantic_set_ids[i] == -1:
                # If response has not been assigned an id, assign it semantic_id.
                semantic_set_ids[i] = semantic_id
                for j in range(i + 1, len(responses)):
                    # Search through all remaining responses. If they are equivalent to response, assign them the same
                    # semantic id.
                    if check_equivalence(
                        response, responses[j]
                    ):  # TODO Ji added strict_entailment=strict_entailment
                        semantic_set_ids[j] = semantic_id
                semantic_id += 1

        assert -1 not in semantic_set_ids
        return semantic_set_ids

    def get_semantic_similarity_matrix(
        self, responses: List[str], method: str = "entailment"
    ) -> np.array:
        """
        Given a list of n responses, returns an n x n matrix K of pairwise similarities.
        """
        if method == "entailment":
            similarity_fn = lambda x, y: self.entailment_model.check_implication(
                x, y, output_probs=True
            )[2]
        else:
            raise NotImplementedError  # TODO
        K = np.eye(len(responses))
        for i, text1 in enumerate(responses):
            for j, text2 in enumerate(responses):
                K[i, j] = similarity_fn(text1, text2)
        return K

    def get_laplacian_matrix(
        self,
        responses: List[str],
        method: str = "entailment",
        symmetrize: bool = True,
        normalize: bool = True,
        return_degree_matrix: bool = True,
    ) -> np.array or (np.array, np.array):
        """
        Computes the Laplacian matrix corresponding to the semantic similarity matrix of responses.
        """
        W = self.get_semantic_similarity_matrix(responses, method)
        if symmetrize:
            W = (W + W.transpose()) / 2.0
        D = np.sum(W, axis=1)
        if normalize:
            D_aux = np.diag(1.0 / np.sqrt(D))
            L = np.eye(len(responses)) - D_aux @ W @ D_aux
        else:
            L = np.diag(D) - W
        if return_degree_matrix:
            return L, np.diag(D)
        return L

    def get_normalized_heat_kernel_from_laplacian(self, L: np.array, t: float = 0.4) -> np.array:
        """
        Given the matrix L corresponding to a graph Laplacian, computes the corresponding kernel matrix using the heat
        kernel with hyperparameter t. The default value of t is taken from https://arxiv.org/pdf/2405.20003.
        """
        K = scipy.linalg.expm(-t * L)
        D_norm = np.diag(1.0 / np.sqrt(np.diagonal(K)))
        return D_norm @ K @ D_norm / K.shape[0]


class SemanticUQ:
    def __init__(
        self, entailment_model="deberta", strict_entailment=False, llm_engine="gpt-3.5-turbo"
    ):
        self.semantic_clustering = SemanticClustering(
            entailment_model, strict_entailment, llm_engine
        )

    def compute_entropy(self, x, log_probs=True, eps=1e-8):
        """
        Computes the entropy of a vector x. If log_probs is true, x corresponds to a vector of log probabilities.
        Otherwise, x corresponds to actual probabilities. In this case eps is used for numerical stability to avoid
        taking the logarithm of 0.
        """
        if log_probs:
            entropy = -np.sum(x * np.exp(x))
        else:
            x_pos = [y for y in x if y > eps]
            entropy = -np.sum(np.log(x_pos) * x_pos)
        return entropy

    def get_semantic_entropy_from_clusters(
        self,
        semantic_set_ids: List[int],
        log_probs: List[List[float]] or None = None,
        length_normalization: bool = True,
    ) -> float:
        """
        Computes semantic entropy from semantic clusters.

        Inputs:
        semantic_set_ids: cluster assignments
        log_probs: log probabilities associated with the generated responses, the outermost list corresponds to
                   responses, and innermost list corresponds to the tokes within that response. These are used to
                   compute cluster probabilities. When log_probs are None, each cluster is assigned probability
                   proportional to its size.
        length_normalization: if True, the log probabilities of each response are taken as the average -- rather than
                              the sum -- of the corresponding token log_probs.

        Outputs:
        semantic entropy
        """
        unique_ids = sorted(list(set(semantic_set_ids)))
        assert unique_ids == list(range(len(unique_ids)))

        # Get response log probs:
        if log_probs is None:
            log_probs_agg = [-np.log(len(semantic_set_ids))] * len(semantic_set_ids)
        else:
            # Get response log probs from token log probs
            if length_normalization:
                log_probs_agg = [np.mean(token_log_probs) for token_log_probs in log_probs]
            else:
                log_probs_agg = [np.sum(token_log_probs) for token_log_probs in log_probs]

        log_probs_per_semantic_id = []
        for uid in unique_ids:
            # Find indices of responses with this semantic id
            id_indices = [pos for pos, x in enumerate(semantic_set_ids) if x == uid]
            # Get log prob for this cluster
            id_log_prob = np.sum([log_probs_agg[i] for i in id_indices])
            log_probs_per_semantic_id.append(id_log_prob)
        if log_probs is None:
            cluster_log_probs = log_probs_per_semantic_id
        else:
            # cluster log probs need not sum to 1, and they should be normalized before computing entropy
            log_norm_const = logsumexp(log_probs_per_semantic_id)
            cluster_log_probs = [
                cluster_log_prob - log_norm_const for cluster_log_prob in log_probs_per_semantic_id
            ]
        return self.compute_entropy(cluster_log_probs)

    def get_semantic_entropy(
        self,
        responses: List[str],
        prompt: str or None = None,
        method: str = "llm",
        log_probs: List[List[float]] or None = None,
        length_normalization: bool = True,
        return_num_sets: bool = False,
    ) -> float or (float, float):
        """
        Computes semantic entropy from given responses.

        Inputs:
        responses: list of responses
        prompt: prompt used to get the responses, optional; this is only relevant when method="llm"
        method: specifies how semantic clustering is done
        log_probs: log probabilities associated with the generated responses, the outermost list corresponds to
                   responses, and innermost list corresponds to the tokes within that response. These are used to
                   compute cluster probabilities. When log_probs are None, each cluster is assigned probability
                   proportional to its size.
        length_normalization: if True, the log probabilities of each response are taken as the average -- rather than
                              the sum -- of the corresponding token log_probs.
        return_num_sets: if True, returns number of semantic clusters in addition to semantic entropy

        Outputs:
        semantic entropy
        """
        semantic_set_ids = self.semantic_clustering.get_semantic_ids(responses, prompt, method)
        semantic_entropy = self.get_semantic_entropy_from_clusters(
            semantic_set_ids, log_probs, length_normalization
        )
        if return_num_sets:
            return semantic_entropy, len(np.unique(semantic_set_ids))
        return semantic_entropy

    def get_laplacian_uncertainties(
        self, responses: List[str], method: str = "entailment"
    ) -> (float, float):
        """
        Computes U_EigV and U_Deg from the paper "Generating with Confidence: Uncertainty Quantification for Black-box
        Large Language Models" by Lin et al., TMLR 2024 (https://arxiv.org/pdf/2305.19187).
        """
        L, D = self.semantic_clustering.get_laplacian_matrix(responses, method)
        eigenvalues, _ = np.linalg.eig(L)
        u_eig_v = np.sum(np.maximum(0.0, 1.0 - np.real(eigenvalues)))
        u_deg = 1.0 - np.trace(D) / (D.shape[0] ** 2)
        return u_eig_v, u_deg

    def get_semantic_von_neumann_entropy(
        self, responses: List[str], method: str = "entailment", t: float = 0.4
    ) -> float:
        """
        Computes the semantic Von Neumann entropy as described in "Kernel Language Entropy: Fine-grained Uncertainty
        Quantification for LLMs from Semantic Similarities" by Nikitin et al. 2024 (https://arxiv.org/pdf/2405.20003).
        Currently only heat kernel is available, which was reported as the best-performing option by Nikitin et al.
        """
        L = self.semantic_clustering.get_laplacian_matrix(
            responses, method, normalize=False, return_degree_matrix=False
        )
        K = self.semantic_clustering.get_normalized_heat_kernel_from_laplacian(L, t)
        eigenvalues, _ = np.linalg.eig(K)
        return self.compute_entropy(np.real(eigenvalues).tolist(), log_probs=False)
