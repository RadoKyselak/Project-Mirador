from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Loads all environment variables into a single, accessible object."""
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    GEMINI_API_KEY: str
    DATA_GOV_API_KEY: str
    BEA_API_KEY: str
    CENSUS_API_KEY: str
    CONGRESS_API_KEY: str
    BLS_API_KEY: str
    FRED_API_KEY: str

    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com"

    @property
    def GEMINI_ENDPOINT(self) -> str:
        return f"{self.GEMINI_BASE_URL}/v1beta/models/{self.GEMINI_MODEL}:generateContent"

settings = Settings()
