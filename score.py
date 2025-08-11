from text2vec import SentenceModel, semantic_search, Similarity


class ScoreModel:
    def __init__(self):
        self.simModel = SentenceModel(model_name_or_path="shibing624/text2vec-base-chinese", device='cuda:0')
    
    def get_score(self, ground_truth_answer, ground_truth_keywords, generated_answer):
        if ground_truth_answer.strip() == "无答案":
            if len(generated_answer) <= 5 and "无答案" in generated_answer:
                return 1.0, 1.0, 1.0
            else:
                return 0.0, 0.0, 0.0
        if len(ground_truth_answer.strip()) > 0 and len(generated_answer.strip()) == 0:
            return 0.0, 0.0, 0.0

        semantic_score = semantic_search(self.simModel.encode([ground_truth_answer]), self.simModel.encode(generated_answer), top_k=1)[0][0]['score']
        recall_keywords = [word for word in ground_truth_keywords if word in generated_answer]
        keyword_score = len(recall_keywords) / (len(ground_truth_keywords) + 1e-5)
        return 0.5*semantic_score+0.5*keyword_score, semantic_score, keyword_score
