from cancer_ai.validator.model_manager import ModelInfo
 
def get_mock_hotkeys_with_models():
    return {
        #good model
        "5HeH6kmR6FyfC6K39aGozMJ3wUTdgxrQAQsy4BBbskxHKqgG": ModelInfo(
            hf_repo_id="eatcats/test",
            hf_model_filename="melanoma-1-piwo.onnx",
            hf_repo_type="model",
        ),
        #wrong model
        "5CQFdhmRyQtiTwHLumywhWtQYTQkF4SpGtdT8aoh3WK3E4E2": ModelInfo(
            hf_repo_id="eatcats/melanoma-test",
            hf_model_filename="2024-08-24_04-37-34-melanoma-1.onnx",
            hf_repo_type="model",
        ),
    }