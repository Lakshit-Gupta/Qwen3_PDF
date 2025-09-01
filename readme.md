# EduPlan AI: Automated MCQ Generator with RAG & Qdrant

## Overview
EduPlan AI is an automated lesson and MCQ generator for educational content. It uses Retrieval-Augmented Generation (RAG) with a Qdrant vector database and OpenAI's GPT models to generate high-quality multiple-choice questions (MCQs) from your curriculum PDFs and processed data.

## Features
- **Semantic Search**: Uses Qdrant vector database for fast, relevant retrieval of curriculum content.
- **MCQ Generation**: Generates exactly 5 MCQs for any topic using OpenAI's GPT-4o.
- **Flexible Filtering**: Filter MCQs by chapter, section, or content type.
- **Embeddings**: Supports NVIDIA Qwen3-Embedding-4B and custom embedding pipelines.
- **Easy Output**: MCQs are saved in `Output_Lesson_Plans` as JSON files for easy use.

## Folder Structure
```
Qwen3_PDF/
├── config.py                # Project configuration (Qdrant host, port, etc.)
├── docker-compose.yml       # Qdrant database container setup
├── embedding_model.py       # Embedding pipeline (Qwen3/NVIDIA)
├── lesson_plan_generator.py # Main MCQ generator script
├── process_improved_data.py # Data processing and embedding script
├── qdrant_connector.py      # Qdrant database connector
├── requirements.txt         # Python dependencies
├── Output_Lesson_Plans/     # Generated MCQ JSON files
├── rag_data/                # Raw and processed curriculum PDFs
│   ├── raw/                 # Original PDFs
│   └── processed/           # Processed data (if any)
```

## Setup
1. **Clone the repository**
2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```
3. **Start Qdrant database**:
   ```sh
   docker-compose up -d
   ```
4. **Configure environment variables**:
   - Create a `.env` file in the project root with your OpenAI API key:
     ```env
     OPENAI_API_KEY=sk-...
     QDRANT_HOST=localhost
     QDRANT_PORT=6333
     QDRANT_COLLECTION_NAME=science_9_collection
     QDRANT_VECTOR_SIZE=2560
     ```

## Usage
### 1. Process Curriculum Data
If you have new or improved curriculum data, run:
```sh
python process_improved_data.py
```
This will generate embeddings and store them in Qdrant.

### 2. Generate MCQs
Run the MCQ generator for any topic:
```sh
python lesson_plan_generator.py --topic "Energy" --chapter "10" --section "10.1"
```
- The generated MCQs will be saved in `Output_Lesson_Plans/MCQs_energy.json`.
- You can omit `--chapter` and `--section` for broader search.

## Output Format
Each MCQ JSON file contains:
- 5 MCQs with question, options (A-D), correct answer, and explanation
- Metadata: topic, chapter, section, source preview

## Customization
- **Embeddings**: You can use your own embedding model by editing `embedding_model.py`.
- **Qdrant Collection**: Change collection name/vector size in `config.py` and `.env`.
- **MCQ Format**: Edit `lesson_plan_generator.py` for custom output formatting.

## Troubleshooting
- Ensure Qdrant is running (`docker-compose up -d`).
- Make sure your `.env` file contains a valid OpenAI API key.
- Check `requirements.txt` for missing dependencies.
- For import errors, run scripts from the project root.

## License
MIT License

## Credits
- [Qdrant Vector Database](https://qdrant.tech/)
- [LangChain](https://langchain.com/)
- [OpenAI GPT](https://platform.openai.com/)
- [NVIDIA Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
