import os


# Unified Language Model Manager
class LLM:
    def __init__(self, model_name: str):
        # Initialize the LLM object with a specific model name
        self.model_name = model_name

        # Automatically identify and assign the corresponding model class
        if "gpt" in model_name:
            self.model = self.ChatGPT(model_name)
        elif "gemini" in model_name:
            self.model = self.Gemini(model_name)
        elif "Llama" in model_name or "WizardLM" in model_name or "Qwen" in model_name:
            self.model = self.DeepInfra(model_name)
        else:
            # Raise an error for unknown model names
            raise ValueError(f"Unknown model: {model_name}")

    def chat(self, user_input: str, temperature: float = 0.0) -> str:
        """
        Unified method to send a chat input to the underlying model and return the response.
        :param user_input: User's input string.
        :param temperature: Sampling temperature for generation.
        :return: Model's response as a string.
        """
        return self.model.chat(user_input, temperature)

    # Subclass for ChatGPT models (using OpenAI API)
    class ChatGPT:
        def __init__(self, model_name: str):
            import logging
            from openai import OpenAI

            # Initialize OpenAI API client using the environment variable for API key
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")
            )
            # Disable OpenAI logging for cleaner output
            logging.getLogger("openai").setLevel(logging.CRITICAL)

            self.model_name = model_name

        def chat(self, user_input: str, temperature: float = 0.0) -> str:
            """
            Send a message to the ChatGPT model and return the generated response.
            """
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": user_input}],
                temperature=temperature,
            )
            return response.choices[0].message.content

    # Subclass for Gemini models (using Google Generative AI API)
    class Gemini:
        def __init__(self, model_name: str):
            import google.generativeai as genai

            # Configure Google Generative AI API client with the API key
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "<your Gemini API key if not set as env var>"))
            self.genai = genai

            self.model_name = model_name

        def chat(self, user_input: str, temperature: float = 0.0) -> str:
            """
            Send a message to the Gemini model and return the generated response.
            """
            model = self.genai.GenerativeModel(
                self.model_name,
                generation_config=self.genai.GenerationConfig(
                    temperature=temperature,
                )
            )
            response = model.generate_content(user_input)
            return response.text

    # Subclass for DeepInfra models (custom API endpoint for Llama, Qwen, WizardLM)
    class DeepInfra:
        def __init__(self, model_name: str):
            from openai import OpenAI
            # Initialize the DeepInfra API client
            api_key = os.environ.get("DEEPINFRA_TOKEN", "<your deepinfra token if not set as env var>")
            self.openai = OpenAI(
                api_key=api_key,
                base_url="https://api.deepinfra.com/v1/openai",
            )

            # Map specific models to their API endpoints
            if 'Llama' in model_name:
                self.model_name = f'meta-llama/{model_name}'
            elif 'Qwen' in model_name:
                self.model_name = f'Qwen/{model_name}'
            elif 'WizardLM' in model_name:
                self.model_name = f'microsoft/{model_name}'
            else:
                self.model_name = model_name

        def chat(self, user_input: str, temperature: float = 0.0) -> str:
            """
            Send a message to the DeepInfra-hosted model and return the generated response.
            """
            chat_completion = self.openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": f"{user_input}"}],
                temperature=temperature,
            )
            return chat_completion.choices[0].message.content


# Predefined list of known models
models_known = [
    "gpt-3.5-turbo-0125",               # GPT model
    "gpt-4o-2024-08-06",                # GPT-4 model
    "gemini-1.5-pro",                   # Gemini model
    "Llama-3.2-90B-Vision-Instruct",    # Llama model
    "WizardLM-2-8x22B",                 # WizardLM model
    "Qwen2.5-72B-Instruct"              # Qwen model
]

if __name__ == '__main__':
    # Examples of using the LLM class to interface with different models
    gpt = LLM('gpt-4o-2024-08-06')      # Initialize a GPT model
    gemini = LLM('gemini-1.5-pro')      # Initialize a Gemini model
    llama = LLM("Llama-3.2-90B-Vision-Instruct")    # Initialize a Llama model
    wizardlm = LLM("WizardLM-2-8x22B")  # Initialize a WizardLM model
    qwen = LLM("Qwen2.5-72B-Instruct")  # Initialize a Qwen model

    # Print responses from different models
    print(f"ChatGPT: {gpt.chat('Hi! Who are you?')}")
    print(f"Gemini: {gemini.chat('Hi! Who are you?')}")
    print(f"LLaMA: {llama.chat('Hi! Who are you?')}")
    print(f"WizardLM: {wizardlm.chat('Hi! Who are you?')}")
    print(f"Qwen: {qwen.chat('Hi! Who are you?')}")
