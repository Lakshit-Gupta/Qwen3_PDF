#!/usr/bin/env python3
"""
Simplified MCQ Generator for EduPlan AI
Generates exactly 5 MCQs for any given topic and saves them in Output_Lesson_Plans
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from langchain_qdrant import Qdrant
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models

# Local imports - adjust these paths according to your project structure
from qdrant_connector import QdrantConnector
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, QDRANT_VECTOR_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCQQuestion(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_answer: str
    explanation: str

class MCQSet(BaseModel):
    mcqs: List[MCQQuestion]

class MCQGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("❌ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.qdrant_connector = QdrantConnector(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=QDRANT_COLLECTION_NAME,
            vector_size=QDRANT_VECTOR_SIZE
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-4B",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = self._initialize_vector_store()
        self.llm = ChatOpenAI(
            model="gpt-4o",  # <-- FIXED MODEL NAME
            temperature=0.3,
            openai_api_key=self.openai_api_key
        )
        self.output_parser = PydanticOutputParser(pydantic_object=MCQSet)
        logger.info("✅ MCQ Generator initialized successfully")

    def _initialize_vector_store(self):
        try:
            vector_store = Qdrant(
                client=self.qdrant_connector.client,
                collection_name=QDRANT_COLLECTION_NAME,
                embeddings=self.embeddings
            )
            logger.info("✅ Connected to Qdrant vector store")
            return vector_store
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            return None

    def retrieve_relevant_content(self, topic: str, chapter: str = None, section: str = None, 
                                content_type: str = None, top_k: int = 8) -> List[Any]:
        search_query = topic
        logger.info(f"🔍 Searching for topic: '{search_query}'")
        query_embedding = self.embeddings.embed_query(search_query)
        must_conditions = []
        if chapter:
            must_conditions.append(models.FieldCondition(key="chapter_number", match=models.MatchValue(value=str(chapter))))
        if section:
            must_conditions.append(models.FieldCondition(key="section_number", match=models.MatchValue(value=str(section))))
        if content_type:
            must_conditions.append(models.FieldCondition(key="content_type", match=models.MatchValue(value=content_type)))
        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            if query_filter:
                logger.info(f"🎯 Applying filters: {must_conditions}")
                results = client.query_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    search_params=models.SearchParams(hnsw_ef=128, exact=False)
                )
            else:
                logger.info("🌐 Searching all documents without filters")
                results = client.query_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query=query_embedding,
                    limit=top_k,
                    search_params=models.SearchParams(hnsw_ef=128, exact=False)
                )
            logger.info(f"✅ Found {len(results.points)} relevant documents via Query API")
            docs = []
            for point in results.points:
                page_content = (point.payload.get("text", "") or 
                              point.payload.get("content", "") or
                              point.payload.get("chunk_text", "") or
                              str(point.payload))
                metadata = point.payload.copy()
                doc = type('Doc', (), {})()
                doc.page_content = page_content
                doc.metadata = metadata
                doc.score = getattr(point, 'score', None)
                docs.append(doc)
            if docs:
                logger.info(f"📄 Sample doc metadata: {docs[0].metadata}")
                logger.info(f"📝 Sample content length: {len(docs[0].page_content)} chars")
            return docs
        except Exception as e:
            logger.error(f"❌ Error retrieving content via Query API: {e}")
            logger.error(f"🔍 Query details - Collection: {QDRANT_COLLECTION_NAME}, Filter: {query_filter}")
            return []

    def generate_mcqs(self, topic: str, chapter: str = None, section: str = None, 
                     content_type: str = None) -> Dict[str, Any]:
        relevant_docs = self.retrieve_relevant_content(topic, chapter, section, content_type)
        if not relevant_docs:
            logger.warning("⚠️ No documents found. Trying without filters...")
            relevant_docs = self.retrieve_relevant_content(topic)
        if not relevant_docs:
            return {
                "error": "No relevant content found in the knowledge base",
                "topic": topic,
                "suggestions": [
                    "Check if Qdrant contains data",
                    "Try a different topic",
                    "Check if embedding model matches the one used for indexing"
                ]
            }
        context = self._extract_context_from_docs(relevant_docs)
        try:
            mcq_set = self._generate_mcqs_with_openai(topic, context)
            return {
                "mcqs": [
                    {
                        "question": mcq.question,
                        "options": {
                            "A": mcq.option_a,
                            "B": mcq.option_b,
                            "C": mcq.option_c,
                            "D": mcq.option_d
                        },
                        "correct_answer": mcq.correct_answer,
                        "explanation": mcq.explanation
                    }
                    for mcq in mcq_set.mcqs
                ],
                "topic": topic,
                "chapter": chapter,
                "section": section,
                "generated_at": datetime.now().isoformat(),
                "source_count": len(relevant_docs),
                "sources_preview": [
                    {
                        "chapter": doc.metadata.get('chapter_title', 'Unknown'),
                        "section": doc.metadata.get('section_title', 'Unknown'),
                        "content_type": doc.metadata.get('content_type', 'Unknown'),
                        "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    }
                    for doc in relevant_docs[:3]
                ]
            }
        except Exception as e:
            logger.error(f"❌ Error generating MCQs: {e}")
            return {
                "error": f"Failed to generate MCQs: {str(e)}",
                "topic": topic,
                "context_found": len(relevant_docs) > 0,
                "context_preview": relevant_docs[0].page_content[:200] if relevant_docs else "No context"
            }

    def _extract_context_from_docs(self, docs: List[Any]) -> str:
        context_parts = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            metadata = doc.metadata
            context_part = f"""
Document {i+1}:
Chapter: {metadata.get('chapter_title', 'Unknown')}
Section: {metadata.get('section_title', 'Unknown')}
Content: {content}
"""
            context_parts.append(context_part)
        return "\n".join(context_parts)

    def _generate_mcqs_with_openai(self, topic: str, context: str) -> MCQSet:
        prompt_template = """
You are an expert educator and assessment designer. Create exactly 5 multiple choice questions (MCQs) based on the provided curriculum content.

CURRICULUM CONTENT:
{context}

REQUIREMENTS:
- Topic: {topic}
- Generate exactly 5 MCQs based on the provided content
- Each question should have 4 options (A, B, C, D)
- Questions should test different levels of understanding (knowledge, comprehension, application, analysis)
- Questions should be clear, unambiguous, and educational
- Provide brief explanations for correct answers
- Ensure questions are directly based on the provided curriculum content
- Avoid trick questions or overly complex language

QUESTION DIFFICULTY LEVELS:
1. Basic knowledge/recall
2. Comprehension/understanding  
3. Application of concepts
4. Analysis/evaluation
5. Synthesis/problem-solving

{format_instructions}
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "topic"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )
        chain = prompt | self.llm | self.output_parser
        result = chain.invoke({
            "context": context,
            "topic": topic
        })
        return result

    def save_mcqs(self, mcqs_data: Dict[str, Any], output_dir: str = "Output_Lesson_Plans"):
        safe_topic = "_".join(mcqs_data['topic'].lower().split())
        out_path = Path(output_dir) / f"MCQs_{safe_topic}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mcqs_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved MCQs to {out_path}")

    def format_mcqs_for_display(self, mcqs_data: Dict[str, Any]) -> str:
        if "error" in mcqs_data:
            error_output = f"❌ Error: {mcqs_data['error']}\n"
            if "context_preview" in mcqs_data:
                error_output += f"📄 Context found: {mcqs_data.get('context_found', False)}\n"
                error_output += f"🔍 Preview: {mcqs_data['context_preview']}\n"
            return error_output
        output = f"\n# MCQs: {mcqs_data['topic']}\n"
        if mcqs_data.get('chapter'):
            output += f"**Chapter:** {mcqs_data['chapter']} | "
        if mcqs_data.get('section'):
            output += f"**Section:** {mcqs_data['section']} | "
        output += f"""**Generated:** {mcqs_data['generated_at'][:19]} | **Sources:** {mcqs_data['source_count']} documents\n\n## 📚 Source Materials Used:\n"""
        for i, source in enumerate(mcqs_data.get('sources_preview', []), 1):
            output += f"{i}. **{source['chapter']}** - {source['section']} ({source['content_type']})\n"
            output += f"   Preview: {source['preview']}\n\n"
        output += "---\n"
        for i, mcq in enumerate(mcqs_data['mcqs'], 1):
            output += f"""
**Question {i}:** {mcq['question']}

A) {mcq['options']['A']}
B) {mcq['options']['B']}  
C) {mcq['options']['C']}
D) {mcq['options']['D']}

**Answer:** {mcq['correct_answer']}
**Explanation:** {mcq['explanation']}

---
"""
        return output.strip()

def main():
    try:
        from dotenv import load_dotenv
        dotenv_path = Path(__file__).parent / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            print(f"🔑 Loaded environment variables from {dotenv_path}")
        else:
            load_dotenv()
            print("🔑 Loaded environment variables from default .env location")
    except ImportError:
        print("⚠️ python-dotenv not installed. Environment variables may not be loaded from .env.")    
    import argparse
    parser = argparse.ArgumentParser(description="Generate 5 MCQs for a topic using OpenAI and save to Output_Lesson_Plans.")
    parser.add_argument("--topic", type=str, required=True, help="Topic for MCQ generation")
    parser.add_argument("--chapter", type=str, help="Chapter number (optional)")
    parser.add_argument("--section", type=str, help="Section number (optional)")
    parser.add_argument("--content_type", type=str, help="Content type filter (optional)")
    args = parser.parse_args()

    generator = MCQGenerator()
    result = generator.generate_mcqs(
        topic=args.topic,
        chapter=args.chapter,
        section=args.section,
        content_type=args.content_type
    )
    generator.save_mcqs(result)
    print(generator.format_mcqs_for_display(result))

if __name__ == "__main__":
    main()
# #!/usr/bin/env python3
# """
# LangChain-based Lesson Plan Generator for EduPlan AI
# Uses RAG (Retrieval-Augmented Generation) with Qdrant vector database
# """

# import os
# import logging
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# from langchain_qdrant import Qdrant
# # LangChain imports
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from langchain_openai import ChatOpenAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.chains.question_answering import load_qa_chain

# # Local imports
# from ..database.qdrant_connector import QdrantConnector
# from ..config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, QDRANT_VECTOR_SIZE

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class LessonPlanStructure(BaseModel):
#     """Structured lesson plan output"""
#     title: str = Field(description="Lesson title")
#     subject: str = Field(description="Subject area")
#     grade_level: str = Field(description="Target grade level")
#     duration: str = Field(description="Lesson duration")
#     objectives: List[str] = Field(description="Learning objectives")
#     materials: List[str] = Field(description="Required materials")
#     introduction: str = Field(description="Introduction activity")
#     main_activities: List[str] = Field(description="Main teaching activities")
#     assessment: str = Field(description="Assessment strategy")
#     differentiation: str = Field(description="Differentiation strategies")
#     standards: List[str] = Field(description="Curriculum standards alignment")


# class LangChainLessonGenerator:
#     """LangChain-powered lesson plan generator using Qdrant embeddings"""

#     def __init__(self, openai_api_key: Optional[str] = None):
#         """Initialize the lesson generator with LangChain components"""

#         # Initialize OpenAI API key
#         self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
#         if not self.openai_api_key:
#             logger.warning("⚠️ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

#         # Initialize Qdrant connection
#         self.qdrant_connector = QdrantConnector(
#             host=QDRANT_HOST,
#             port=QDRANT_PORT,
#             collection_name=QDRANT_COLLECTION_NAME,
#             vector_size=QDRANT_VECTOR_SIZE
#         )
        

#         # Initialize embeddings (using same model as your embeddings)
#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="Qwen/Qwen3-Embedding-4B",
#             model_kwargs={'device': 'cpu'},  # Use CPU for retrieval
#             encode_kwargs={'normalize_embeddings': True}
#         )

#         # Initialize LangChain Qdrant vector store
#         self.vector_store = self._initialize_vector_store()

        
#         # Initialize LLM
#         self.llm = ChatOpenAI(
#             model="gpt-4o-mini",  # Cost-effective model
#             temperature=0.3,
#             openai_api_key=self.openai_api_key
#         ) if self.openai_api_key else None

#         # Initialize output parser
#         self.output_parser = PydanticOutputParser(pydantic_object=LessonPlanStructure)

#         logger.info("✅ LangChain Lesson Generator initialized")

#     def _initialize_vector_store(self):
#         """Initialize LangChain Qdrant vector store"""
#         try:
#             vector_store = Qdrant(
#                 client=self.qdrant_connector.client,
#                 collection_name=QDRANT_COLLECTION_NAME,
#                 embeddings=self.embeddings
#             )
#             logger.info("✅ Connected to Qdrant vector store")
#             return vector_store
#         except Exception as e:
#             logger.error(f"❌ Failed to initialize vector store: {e}")
#             return None

#     def retrieve_relevant_content(self, topic: str, grade_level: str = None, subject: str = None, top_k: int = 8) -> List[Any]:
#         """Retrieve relevant educational content from Qdrant"""

#         # Build search query
#         search_query = f"{topic}"
#         if grade_level:
#             search_query += f" grade {grade_level}"
#         if subject:
#             search_query += f" {subject}"

#         logger.info(f"🔍 Searching for: '{search_query}'")

#         try:
#             # Perform similarity search
#             docs = self.vector_store.similarity_search(
#                 query=search_query,
#                 k=top_k,
#                 filter=None
#             )

#             logger.info(f"✅ Found {len(docs)} relevant documents")
#             return docs

#         except Exception as e:
#             logger.error(f"❌ Error retrieving content: {e}")
#             return []
#         def retrieve_relevant_content(self, topic: str, grade_level: str = None, subject: str = None, chapter: str = None, section: str = None, top_k: int = 8) -> List[Any]:
#             """Retrieve relevant educational content from Qdrant using Query API"""

#             from qdrant_client import QdrantClient, models
#             from langchain_community.embeddings import HuggingFaceEmbeddings

#             # Build search query
#             search_query = f"{topic}"
#             if grade_level:
#                 search_query += f" grade {grade_level}"
#             if subject:
#                 search_query += f" {subject}"
#             if chapter:
#                 search_query += f" chapter {chapter}"
#             if section:
#                 search_query += f" section {section}"

#             logger.info(f"🔍 Searching for: '{search_query}'")

#             # Get embedding for query
#             query_embedding = self.embeddings.embed_query(search_query)

#             # Build advanced filter
#             must_conditions = []
#             if grade_level:
#                 must_conditions.append(models.FieldCondition(key="grade_level", match=models.MatchValue(value=grade_level)))
#             if subject:
#                 must_conditions.append(models.FieldCondition(key="subject", match=models.MatchValue(value=subject)))
#             if chapter:
#                 must_conditions.append(models.FieldCondition(key="chapter_number", match=models.MatchValue(value=chapter)))
#             if section:
#                 must_conditions.append(models.FieldCondition(key="section_number", match=models.MatchValue(value=section)))

#             query_filter = models.Filter(must=must_conditions) if must_conditions else None

#             # Use QdrantClient Query API
#             try:
#                 client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
#                 results = client.query_points(
#                     collection_name=QDRANT_COLLECTION_NAME,
#                     query=query_embedding,
#                     query_filter=query_filter,
#                     limit=top_k,
#                     search_params=models.SearchParams(hnsw_ef=128, exact=False)
#                 )
#                 logger.info(f"✅ Found {len(results)} relevant documents via Query API")
#                 # Convert results to LangChain Document-like objects
#                 docs = []
#                 for point in results:
#                     page_content = point.payload.get("text", "")
#                     metadata = point.payload.copy()
#                     doc = type('Doc', (), {})()
#                     doc.page_content = page_content
#                     doc.metadata = metadata
#                     doc.score = getattr(point, 'score', None)
#                     docs.append(doc)
#                 return docs
#             except Exception as e:
#                 logger.error(f"❌ Error retrieving content via Query API: {e}")
#                 return []

#     def _build_filter(self, grade_level: str = None, subject: str = None) -> Optional[Dict[str, Any]]:
#         """Build Qdrant filter for search"""
#         filter_conditions = []

#         if grade_level:
#             filter_conditions.append({
#                 "key": "metadata.grade_level",
#                 "match": {"value": grade_level}
#             })

#         if subject:
#             filter_conditions.append({
#                 "key": "metadata.subject",
#                 "match": {"value": subject}
#             })

#         if filter_conditions:
#             return {"must": filter_conditions}
#         return None
#         # _build_filter is now handled by Query API above

#     def generate_lesson_plan(self, topic: str, grade_level: str = "9-12", subject: str = "General",
#                            duration: str = "45 minutes", custom_requirements: str = "", chapter: str = None, section: str = None) -> Dict[str, Any]:
#         """
#         Generate a comprehensive lesson plan using RAG

#         Args:
#             topic: Main lesson topic
#             grade_level: Target grade level
#             subject: Subject area
#             duration: Lesson duration
#             custom_requirements: Additional requirements or constraints
#             chapter: Optional chapter number for filtering
#             section: Optional section number for filtering
#         """
#         # Retrieve relevant content
#         relevant_docs = self.retrieve_relevant_content(topic, grade_level, subject, chapter, section)

#         if not relevant_docs:
#             return {
#                 "error": "No relevant content found in the knowledge base",
#                 "topic": topic,
#                 "suggestions": ["Try a different topic", "Check grade level", "Verify subject area"]
#             }

#         # Extract context from retrieved documents
#         context = self._extract_context_from_docs(relevant_docs)

#         # Generate lesson plan using LangChain
#         if self.llm:
#             lesson_plan = self._generate_with_langchain(topic, context, grade_level, subject, duration, custom_requirements)
#         else:
#             lesson_plan = self._generate_fallback_lesson_plan(topic, context, grade_level, subject, duration)

#         return {
#             "lesson_plan": lesson_plan,
#             "sources": [
#                 {
#                     "content": doc.page_content[:200] + "...",
#                     "metadata": doc.metadata,
#                     "score": getattr(doc, 'score', None)
#                 } for doc in relevant_docs
#             ],
#             "topic": topic,
#             "grade_level": grade_level,
#             "subject": subject,
#             "generated_at": datetime.now().isoformat(),
#             "source_count": len(relevant_docs)
#         }

#         if not relevant_docs:
#             return {
#                 "error": "No relevant content found in the knowledge base",
#                 "topic": topic,
#                 "suggestions": ["Try a different topic", "Check grade level", "Verify subject area"]
#             }

#         # Extract context from retrieved documents
#         context = self._extract_context_from_docs(relevant_docs)

#         # Generate lesson plan using LangChain
#         if self.llm:
#             lesson_plan = self._generate_with_langchain(topic, context, grade_level, subject, duration, custom_requirements)
#         else:
#             lesson_plan = self._generate_fallback_lesson_plan(topic, context, grade_level, subject, duration)

#         return {
#             "lesson_plan": lesson_plan,
#             "sources": [
#                 {
#                     "content": doc.page_content[:200] + "...",
#                     "metadata": doc.metadata,
#                     "score": getattr(doc, 'score', None)
#                 } for doc in relevant_docs
#             ],
#             "topic": topic,
#             "grade_level": grade_level,
#             "subject": subject,
#             "generated_at": datetime.now().isoformat(),
#             "source_count": len(relevant_docs)
#         }

#     def _extract_context_from_docs(self, docs: List[Any]) -> str:
#         """Extract and format context from retrieved documents"""
#         context_parts = []

#         for i, doc in enumerate(docs):
#             content = doc.page_content.strip()
#             metadata = doc.metadata

#             # Format context with metadata
#             context_part = f"""
# Document {i+1}:
# Chapter: {metadata.get('chapter_title', 'Unknown')}
# Section: {metadata.get('section_title', 'Unknown')}
# Content: {content}
# """
#             context_parts.append(context_part)

#         return "\n".join(context_parts)

#     def _generate_with_langchain(self, topic: str, context: str, grade_level: str,
#                                subject: str, duration: str, custom_requirements: str) -> str:
#         """Generate lesson plan using LangChain and LLM"""

#         # Create prompt template
#         prompt_template = """
# You are an expert educational curriculum designer. Create a comprehensive, standards-aligned lesson plan using the provided context.

# CONTEXT FROM CURRICULUM:
# {context}

# LESSON REQUIREMENTS:
# - Topic: {topic}
# - Grade Level: {grade_level}
# - Subject: {subject}
# - Duration: {duration}
# - Additional Requirements: {custom_requirements}

# Generate a detailed lesson plan that includes:
# 1. Clear learning objectives
# 2. Engaging introduction activity
# 3. Main teaching activities with timing
# 4. Hands-on practice activities
# 5. Assessment strategies
# 6. Differentiation for diverse learners
# 7. Required materials
# 8. Curriculum standards alignment
# 9. Extension activities

# Format the lesson plan professionally with clear sections and actionable details.
# Ensure the plan is age-appropriate for {grade_level} students and aligns with {subject} curriculum standards.

# {format_instructions}
# """

#         prompt = PromptTemplate(
#             template=prompt_template,
#             input_variables=["context", "topic", "grade_level", "subject", "duration", "custom_requirements"],
#             partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
#         )

#         try:
#             # Create chain
#             chain = prompt | self.llm | self.output_parser

#             # Generate lesson plan
#             result = chain.invoke({
#                 "context": context,
#                 "topic": topic,
#                 "grade_level": grade_level,
#                 "subject": subject,
#                 "duration": duration,
#                 "custom_requirements": custom_requirements
#             })

#             # Format as readable lesson plan
#             return self._format_lesson_plan_output(result)

#         except Exception as e:
#             logger.error(f"❌ Error generating lesson plan with LangChain: {e}")
#             return self._generate_fallback_lesson_plan(topic, context, grade_level, subject, duration)

#     def _format_lesson_plan_output(self, structured_plan: LessonPlanStructure) -> str:
#         """Format structured lesson plan into readable text"""

#         lesson_plan = f"""
# # {structured_plan.title}

# ## 📚 Course Information
# - **Subject:** {structured_plan.subject}
# - **Grade Level:** {structured_plan.grade_level}
# - **Duration:** {structured_plan.duration}

# ## 🎯 Learning Objectives
# {chr(10).join(f"{i+1}. {obj}" for i, obj in enumerate(structured_plan.objectives))}

# ## 📋 Materials Required
# {chr(10).join(f"- {material}" for material in structured_plan.materials)}

# ## 🚀 Introduction Activity ({structured_plan.duration})
# {structured_plan.introduction}

# ## 📖 Main Teaching Activities
# {chr(10).join(f"### Activity {i+1}{chr(10)}{activity}" for i, activity in enumerate(structured_plan.main_activities))}

# ## ✅ Assessment Strategy
# {structured_plan.assessment}

# ## 🎭 Differentiation Strategies
# {structured_plan.differentiation}

# ## 📏 Curriculum Standards Alignment
# {chr(10).join(f"- {standard}" for standard in structured_plan.standards)}

# ## 🔄 Extension Activities
# - Advanced practice problems
# - Research projects on related topics
# - Real-world application assignments
# - Peer teaching opportunities

# ---
# *Generated by EduPlan AI - LangChain RAG System*
# *Based on curriculum embeddings from Qdrant vector database*
# """

#         return lesson_plan.strip()

#     def _generate_fallback_lesson_plan(self, topic: str, context: str, grade_level: str,
#                                      subject: str, duration: str) -> str:
#         """Fallback lesson plan generation without LLM"""

#         lesson_plan = f"""
# # Lesson Plan: {topic}

# ## 📚 Course Information
# - **Subject:** {subject}
# - **Grade Level:** {grade_level}
# - **Duration:** {duration}

# ## 🎯 Learning Objectives
# 1. Understand the fundamental concepts of {topic}
# 2. Apply learned principles to solve related problems
# 3. Demonstrate comprehension through structured activities
# 4. Connect new knowledge to existing curriculum

# ## 📋 Materials Required
# - Textbook and reference materials
# - Whiteboard/markers or presentation tools
# - Worksheets and practice exercises
# - Assessment materials

# ## 🚀 Introduction Activity (10 minutes)
# - Engage students with a relevant real-world example
# - Connect to previous knowledge
# - Present learning objectives
# - Set expectations for the lesson

# ## 📖 Main Teaching Activities

# ### Direct Instruction (15 minutes)
# - Present core concepts with clear examples
# - Use visual aids and demonstrations
# - Encourage student questions and participation

# ### Guided Practice (15 minutes)
# - Work through examples together as a class
# - Provide step-by-step guidance
# - Address common misconceptions

# ### Independent Practice (10 minutes)
# - Students work individually on practice problems
# - Circulate to provide individual support
# - Monitor understanding and progress

# ## ✅ Assessment Strategy
# - Formative assessment through observation and questioning
# - Exit ticket with key concept check
# - Homework assignment for reinforcement

# ## 🎭 Differentiation Strategies
# - Provide additional support for struggling students
# - Offer extension activities for advanced learners
# - Use flexible grouping based on readiness

# ## 📏 Curriculum Standards Alignment
# - Aligned with {subject} curriculum standards
# - Meets grade {grade_level} learning expectations
# - Supports progression to next concepts

# ---
# *Generated by EduPlan AI - Fallback Mode*
# *Note: For enhanced lesson plans, configure OpenAI API key*
# """

#         return lesson_plan.strip()

#     def search_similar_topics(self, topic: str, top_k: int = 5) -> List[Dict[str, Any]]:
#         """Search for similar topics in the knowledge base"""
#         try:
#             docs = self.vector_store.similarity_search(topic, k=top_k)
#             return [
#                 {
#                     "topic": doc.metadata.get("section_title", "Unknown"),
#                     "chapter": doc.metadata.get("chapter_title", "Unknown"),
#                     "content_preview": doc.page_content[:150] + "...",
#                     "similarity_score": getattr(doc, 'score', None)
#                 } for doc in docs
#             ]
#         except Exception as e:
#             logger.error(f"❌ Error searching similar topics: {e}")
#             return []


# def main():
#     """Example usage of the LangChain lesson generator"""

#     # Initialize generator
#     generator = LangChainLessonGenerator()

#     # Example lesson generation
#     topic = "Introduction to Atomic Theory"
#     grade_level = "9-10"
#     subject = "Chemistry"

#     print(f"🎓 Generating lesson plan for: {topic}")
#     print(f"📚 Grade Level: {grade_level} | Subject: {subject}")
#     print("-" * 60)

#     result = generator.generate_lesson_plan(
#         topic=topic,
#         grade_level=grade_level,
#         subject=subject,
#         duration="50 minutes"
#     )

#     if "error" in result:
#         print(f"❌ Error: {result['error']}")
#         print("💡 Suggestions:")
#         for suggestion in result.get("suggestions", []):
#             print(f"   - {suggestion}")
#     else:
#         print(result["lesson_plan"])
#         print(f"\n📊 Sources used: {result['source_count']} documents")
#         print(f"⏰ Generated at: {result['generated_at']}")


# if __name__ == "__main__":
#     main()
