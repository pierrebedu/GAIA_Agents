from dotenv import load_dotenv

from tools import transcribe_mp3

load_dotenv()

result = transcribe_mp3.invoke({
    "file_path": "/home/pbedu/gaia_agent/gaia_files/99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3",
})
print(result)
