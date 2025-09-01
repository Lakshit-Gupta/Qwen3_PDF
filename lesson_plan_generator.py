#!/usr/bin/env python3
"""
LangChain-based Lesson Plan Generator for EduPlan AI
Uses RAG (Retrieval-Augmented Generation) with Qdrant vector database
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_qdrant import Qdrant
# LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain

# Local imports
from ..database.qdrant_connector import QdrantConnector
from ..config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, QDRANT_VECTOR_SIZE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LessonPlanStructure(BaseModel):
    """Structured lesson plan output"""
    title: str = Field(description="Lesson title")
    subject: str = Field(description="Subject area")
    grade_level: str = Field(description="Target grade level")
    duration: str = Field(description="Lesson duration")
    objectives: List[str] = Field(description="Learning objectives")
    materials: List[str] = Field(description="Required materials")
    introduction: str = Field(description="Introduction activity")
    main_activities: List[str] = Field(description="Main teaching activities")
    assessment: str = Field(description="Assessment strategy")
    differentiation: str = Field(description="Differentiation strategies")
    standards: List[str] = Field(description="Curriculum standards alignment")


class LangChainLessonGenerator:
    """LangChain-powered lesson plan generator using Qdrant embeddings"""

    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the lesson generator with LangChain components"""

        # Initialize OpenAI API key
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("⚠️ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        # Initialize Qdrant connection
        self.qdrant_connector = QdrantConnector(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            collection_name=QDRANT_COLLECTION_NAME,
            vector_size=QDRANT_VECTOR_SIZE
        )
        

        # Initialize embeddings (using same model as your embeddings)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Qwen/Qwen3-Embedding-4B",
            model_kwargs={'device': 'cpu'},  # Use CPU for retrieval
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize LangChain Qdrant vector store
        self.vector_store = self._initialize_vector_store()

        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # Cost-effective model
            temperature=0.3,
            openai_api_key=self.openai_api_key
        ) if self.openai_api_key else None

        # Initialize output parser
        self.output_parser = PydanticOutputParser(pydantic_object=LessonPlanStructure)

        logger.info("✅ LangChain Lesson Generator initialized")

    def _initialize_vector_store(self):
        """Initialize LangChain Qdrant vector store"""
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
        class MCQGenerator:
            """MCQ Generator using OpenAI API"""
            def __init__(self):
                from dotenv import load_dotenv
                load_dotenv()
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment.")
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)

            def generate_mcqs(self, topic: str, num_mcqs: int = 5) -> List[dict]:
                prompt = (
                    f"Generate {num_mcqs} multiple choice questions (MCQs) for the topic: '{topic}'. "
                    "Each MCQ should have 4 options and indicate the correct answer. "
                    "Return the output as a JSON list with each item containing 'question', 'options', and 'answer'. "
                    "Do not include any explanation or extra text."
                )
                logger.info(f"Requesting MCQs for topic: {topic}")
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800
                )
                content = response.choices[0].message.content
                try:
                    mcqs = json.loads(content)
                except Exception:
                    import re
                    match = re.search(r'\[.*\]', content, re.DOTALL)
                    if match:
                        mcqs = json.loads(match.group(0))
                    else:
                        logger.error("Could not parse MCQs from OpenAI response.")
                        mcqs = []
                return mcqs

            def save_mcqs(self, mcqs: List[dict], topic: str, output_dir: str = "Output_Lesson_Plans"):
                safe_topic = "_".join(topic.lower().split())
                out_path = Path(output_dir) / f"MCQs_{safe_topic}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(mcqs, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved MCQs to {out_path}")
                    "metadata": doc.metadata,
                    "score": getattr(doc, 'score', None)
                } for doc in relevant_docs
            ],
            "topic": topic,
            "grade_level": grade_level,
            "subject": subject,
            "generated_at": datetime.now().isoformat(),
            "source_count": len(relevant_docs)
        }

    def _extract_context_from_docs(self, docs: List[Any]) -> str:
        """Extract and format context from retrieved documents"""
        context_parts = []

        for i, doc in enumerate(docs):
            content = doc.page_content.strip()
            metadata = doc.metadata

            # Format context with metadata
            context_part = f"""
Document {i+1}:
Chapter: {metadata.get('chapter_title', 'Unknown')}
Section: {metadata.get('section_title', 'Unknown')}
Content: {content}
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _generate_with_langchain(self, topic: str, context: str, grade_level: str,
                               subject: str, duration: str, custom_requirements: str) -> str:
        """Generate lesson plan using LangChain and LLM"""

        # Create prompt template
        prompt_template = """
You are an expert educational curriculum designer. Create a comprehensive, standards-aligned lesson plan using the provided context.

CONTEXT FROM CURRICULUM:
{context}

LESSON REQUIREMENTS:
- Topic: {topic}
- Grade Level: {grade_level}
- Subject: {subject}
- Duration: {duration}
- Additional Requirements: {custom_requirements}

Generate a detailed lesson plan that includes:
1. Clear learning objectives
2. Engaging introduction activity
3. Main teaching activities with timing
4. Hands-on practice activities
5. Assessment strategies
6. Differentiation for diverse learners
7. Required materials
8. Curriculum standards alignment
9. Extension activities

Format the lesson plan professionally with clear sections and actionable details.
Ensure the plan is age-appropriate for {grade_level} students and aligns with {subject} curriculum standards.

{format_instructions}
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "topic", "grade_level", "subject", "duration", "custom_requirements"],
            partial_variables={"format_instructions": self.output_parser.get_format_instructions()}
        )

        try:
            # Create chain
            chain = prompt | self.llm | self.output_parser

            # Generate lesson plan
            result = chain.invoke({
                "context": context,
                "topic": topic,
                "grade_level": grade_level,
                "subject": subject,
                "duration": duration,
                "custom_requirements": custom_requirements
            })

            # Format as readable lesson plan
            return self._format_lesson_plan_output(result)

        except Exception as e:
            logger.error(f"❌ Error generating lesson plan with LangChain: {e}")
            return self._generate_fallback_lesson_plan(topic, context, grade_level, subject, duration)

    def _format_lesson_plan_output(self, structured_plan: LessonPlanStructure) -> str:
        """Format structured lesson plan into readable text"""

        lesson_plan = f"""
# {structured_plan.title}

## 📚 Course Information
- **Subject:** {structured_plan.subject}
- **Grade Level:** {structured_plan.grade_level}
- **Duration:** {structured_plan.duration}

## 🎯 Learning Objectives
{chr(10).join(f"{i+1}. {obj}" for i, obj in enumerate(structured_plan.objectives))}

## 📋 Materials Required
{chr(10).join(f"- {material}" for material in structured_plan.materials)}

## 🚀 Introduction Activity ({structured_plan.duration})
{structured_plan.introduction}

## 📖 Main Teaching Activities
{chr(10).join(f"### Activity {i+1}{chr(10)}{activity}" for i, activity in enumerate(structured_plan.main_activities))}

## ✅ Assessment Strategy
{structured_plan.assessment}

## 🎭 Differentiation Strategies
{structured_plan.differentiation}

## 📏 Curriculum Standards Alignment
{chr(10).join(f"- {standard}" for standard in structured_plan.standards)}

## 🔄 Extension Activities
- Advanced practice problems
- Research projects on related topics
- Real-world application assignments
- Peer teaching opportunities

---
*Generated by EduPlan AI - LangChain RAG System*
*Based on curriculum embeddings from Qdrant vector database*
"""

        return lesson_plan.strip()

    def _generate_fallback_lesson_plan(self, topic: str, context: str, grade_level: str,
                                     subject: str, duration: str) -> str:
        """Fallback lesson plan generation without LLM"""

        lesson_plan = f"""
# Lesson Plan: {topic}

## 📚 Course Information
- **Subject:** {subject}
- **Grade Level:** {grade_level}
- **Duration:** {duration}

## 🎯 Learning Objectives
1. Understand the fundamental concepts of {topic}
2. Apply learned principles to solve related problems
3. Demonstrate comprehension through structured activities
4. Connect new knowledge to existing curriculum

## 📋 Materials Required
- Textbook and reference materials
- Whiteboard/markers or presentation tools
- Worksheets and practice exercises
- Assessment materials

## 🚀 Introduction Activity (10 minutes)
- Engage students with a relevant real-world example
- Connect to previous knowledge
- Present learning objectives
- Set expectations for the lesson

## 📖 Main Teaching Activities

### Direct Instruction (15 minutes)
- Present core concepts with clear examples
- Use visual aids and demonstrations
- Encourage student questions and participation

### Guided Practice (15 minutes)
- Work through examples together as a class
- Provide step-by-step guidance
- Address common misconceptions

### Independent Practice (10 minutes)
- Students work individually on practice problems
- Circulate to provide individual support
- Monitor understanding and progress

## ✅ Assessment Strategy
- Formative assessment through observation and questioning
- Exit ticket with key concept check
- Homework assignment for reinforcement

## 🎭 Differentiation Strategies
- Provide additional support for struggling students
- Offer extension activities for advanced learners
- Use flexible grouping based on readiness

## 📏 Curriculum Standards Alignment
- Aligned with {subject} curriculum standards
- Meets grade {grade_level} learning expectations
- Supports progression to next concepts

---
*Generated by EduPlan AI - Fallback Mode*
*Note: For enhanced lesson plans, configure OpenAI API key*
"""

        return lesson_plan.strip()

    def search_similar_topics(self, topic: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar topics in the knowledge base"""
        try:
            docs = self.vector_store.similarity_search(topic, k=top_k)
            return [
                {
                    "topic": doc.metadata.get("section_title", "Unknown"),
                    "chapter": doc.metadata.get("chapter_title", "Unknown"),
                    "content_preview": doc.page_content[:150] + "...",
                    "similarity_score": getattr(doc, 'score', None)
                } for doc in docs
            ]
        except Exception as e:
            logger.error(f"❌ Error searching similar topics: {e}")
            return []


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate 5 MCQs for a topic using OpenAI and save to Output_Lesson_Plans.")
    parser.add_argument("--topic", type=str, required=True, help="Topic for MCQ generation")
    args = parser.parse_args()

    generator = MCQGenerator()
    mcqs = generator.generate_mcqs(args.topic, num_mcqs=5)
    generator.save_mcqs(mcqs, args.topic)
    print(f"✅ MCQs for topic '{args.topic}' generated and saved.")


if __name__ == "__main__":
    main()
