from cancer_ai.validator.model_manager import ModelInfo
 
def get_mock_hotkeys_with_models():
    return {
        "5HeH6kmR6FyfC6K39aGozMJ3wUTdgxrQAQsy4BBbskxHKqgG": ModelInfo(
            hf_repo_id="mock/tricorder3",
            hf_model_filename="sample_tricorder_3_model.onnx",
            hf_repo_type="model",
        ),
        "5CQFdhmRyQtiTwHLumywhWtQYTQkF4SpGtdT8aoh3WK3E4E2": ModelInfo(
            hf_repo_id="mock/tricorder3",
            hf_model_filename="sample_tricorder_3_model.onnx",
            hf_repo_type="model",
        ),
    }