"""
RAG Query System - Question Answering with Retrieval-Augmented Generation
"""
import sys
import warnings
from pathlib import Path

# Suppress LangChain deprecation warnings for HuggingFace classes
# We're using langchain-huggingface package which is the recommended replacement
warnings.filterwarnings('ignore', category=DeprecationWarning, module='langchain')

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    # Fallback for older langchain versions
    from langchain.vectorstores import FAISS

# Try new langchain-huggingface package first, then fallback
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    _using_new_hf_embeddings = True
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _using_new_hf_embeddings = False
    except ImportError:
        # Fallback for older langchain versions
        from langchain.embeddings import HuggingFaceEmbeddings
        _using_new_hf_embeddings = False

from typing import List, Dict, Optional
import pickle

from Scripts.utils import load_config

class RAGSystem:
    """
    RAG-based Question Answering System
    """
    
    def __init__(self, vectorstore_path: str = "Models/rag_vectorstore",
                 embedding_model: str = None,
                 llm_model: str = None,
                 use_rag: bool = True):
        """
        Initialize RAG system
        
        Args:
            vectorstore_path: Path to FAISS vector store
            embedding_model: Embedding model name
            llm_model: LLM model name (or None for baseline)
            use_rag: Whether to use RAG (False for baseline)
        """
        self.vectorstore_path = Path(vectorstore_path)
        self.use_rag = use_rag
        self.vectorstore = None
        self.qa_chain = None
        
        # Load config
        config = load_config()
        rag_config = config['rag']
        
        self.embedding_model = embedding_model or rag_config.get('embedding_model')
        self.llm_model = llm_model or rag_config.get('llm_model')
        self.top_k = rag_config.get('top_k', 5)
        
        if use_rag:
            self._load_vectorstore()
            self._setup_qa_chain()
        else:
            # Baseline: LLM without retrieval
            self._setup_baseline_chain()
    
    def _load_vectorstore(self):
        """Load FAISS vector store"""
        if not self.vectorstore_path.exists():
            raise FileNotFoundError(f"Vector store not found: {self.vectorstore_path}")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        self.vectorstore = FAISS.load_local(
            str(self.vectorstore_path), 
            embeddings,
            allow_dangerous_deserialization=True  # Safe since we created the vectorstore
        )
        print(f"Loaded vector store from {self.vectorstore_path}")
    
    def _setup_qa_chain(self):
        """Set up RAG QA chain - using simple retrieval + LLM approach"""
        # Custom prompt template
        self.prompt_template = """Use the following pieces of context to answer the question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always cite the source document and page number when available.

Context:
{context}

Question: {question}

Answer:"""
        
        # Initialize LLM (using OpenAI API - user needs to set OPENAI_API_KEY)
        # For open source alternatives, use HuggingFacePipeline
        try:
            try:
                from langchain_community.llms import OpenAI
            except ImportError:
                from langchain.llms import OpenAI
            self.llm = OpenAI(temperature=0, model_name=self.llm_model)
            print("Using OpenAI LLM")
        except:
            print("Warning: OpenAI API key not set. Using HuggingFace model...")
            # Try new langchain-huggingface package first, then fallback
            try:
                from langchain_huggingface import HuggingFacePipeline
            except ImportError:
                try:
                    from langchain_community.llms import HuggingFacePipeline
                except ImportError:
                    try:
                        from langchain import HuggingFacePipeline
                    except ImportError:
                        # Use transformers directly
                        from transformers import pipeline
                        self.llm = pipeline(
                            "text-generation",
                            model="gpt2",
                            max_new_tokens=200,
                            temperature=0.7
                        )
                        self.llm_type = "pipeline"
                        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
                        return
            
            from transformers import pipeline
            
            # Use a smaller model for local inference
            pipe = pipeline(
                "text-generation",
                model="gpt2",  # Replace with better model if available
                max_new_tokens=200,
                temperature=0.7
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.llm_type = "huggingface"
        
        # Use simple retrieval approach (no RetrievalQA chain needed)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        self.qa_chain = None  # Not using RetrievalQA chain
    
    def _setup_baseline_chain(self):
        """Set up baseline LLM chain (without RAG)"""
        try:
            try:
                from langchain_community.llms import OpenAI
            except ImportError:
                from langchain.llms import OpenAI
            self.llm = OpenAI(temperature=0, model_name=self.llm_model)
            self.llm_type = "openai"
        except:
            # Try new langchain-huggingface package first, then fallback
            try:
                from langchain_huggingface import HuggingFacePipeline
            except ImportError:
                try:
                    from langchain_community.llms import HuggingFacePipeline
                except ImportError:
                    try:
                        from langchain import HuggingFacePipeline
                    except ImportError:
                        from transformers import pipeline
                        self.llm = pipeline(
                            "text-generation",
                            model="gpt2",
                            max_new_tokens=200,
                            temperature=0.7
                        )
                        self.llm_type = "pipeline"
                        return
            
            from transformers import pipeline
            
            pipe = pipeline(
                "text-generation",
                model="gpt2",
                max_new_tokens=200,
                temperature=0.7
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)
            self.llm_type = "huggingface"
        
        self.qa_chain = None
    
    def query(self, question: str, language: Optional[str] = None) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: Question string (supports English, Urdu, Roman-Urdu)
            language: Optional language hint ("en", "ur", "roman_urdu")
            
        Returns:
            dict: Answer with sources
        """
        if self.use_rag:
            # Retrieve relevant documents
            try:
                docs = self.retriever.get_relevant_documents(question)
            except AttributeError:
                # Newer langchain versions use invoke
                docs = self.retriever.invoke(question)
            
            # Combine context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format prompt
            formatted_prompt = self.prompt_template.format(context=context, question=question)
            
            # Generate answer based on LLM type
            if self.llm_type == "pipeline":
                # Direct transformers pipeline
                result = self.llm(formatted_prompt, return_full_text=False)
                if isinstance(result, list) and len(result) > 0:
                    answer = result[0].get('generated_text', str(result))
                else:
                    answer = str(result)
            elif self.llm_type == "huggingface":
                # HuggingFacePipeline
                try:
                    if hasattr(self.llm, 'invoke'):
                        answer = self.llm.invoke(formatted_prompt)
                    elif hasattr(self.llm, 'predict'):
                        answer = self.llm.predict(formatted_prompt)
                    else:
                        answer = self.llm(formatted_prompt)
                    
                    if isinstance(answer, str):
                        pass
                    elif hasattr(answer, 'get'):
                        answer = answer.get('generated_text', str(answer))
                    else:
                        answer = str(answer)
                except Exception as e:
                    # Try alternative methods
                    try:
                        answer = self.llm.invoke(formatted_prompt) if hasattr(self.llm, 'invoke') else str(self.llm)
                    except:
                        answer = f"Error generating answer: {str(e)}"
            else:
                # OpenAI or other langchain LLM
                try:
                    answer = self.llm(formatted_prompt)
                    if isinstance(answer, str):
                        pass
                    elif hasattr(answer, 'content'):
                        answer = answer.content
                    else:
                        answer = str(answer)
                except:
                    answer = str(self.llm(formatted_prompt))
            
            return {
                "answer": answer.strip(),
                "sources": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "source": doc.metadata.get("source", "unknown"),
                        "page": doc.metadata.get("page", "N/A")
                    }
                    for doc in docs
                ]
            }
        else:
            # Baseline: direct LLM response
            if self.llm_type == "pipeline":
                result = self.llm(question, return_full_text=False)
                answer = result[0].get('generated_text', str(result)) if isinstance(result, list) else str(result)
            elif self.llm_type == "huggingface":
                if hasattr(self.llm, 'invoke'):
                    answer = str(self.llm.invoke(question))
                elif hasattr(self.llm, 'predict'):
                    answer = str(self.llm.predict(question))
                else:
                    answer = str(self.llm)
            else:
                answer = str(self.llm(question))
            
            return {
                "answer": answer.strip(),
                "sources": []  # No sources for baseline
            }

def evaluate_rag_qa(qa_pairs: List[Dict], rag_system: RAGSystem) -> Dict:
    """
    Evaluate RAG system on QA pairs
    
    Args:
        qa_pairs: List of dicts with 'question', 'answer', 'source' keys
        rag_system: RAGSystem instance
        
    Returns:
        dict: Evaluation results
    """
    results = []
    
    for i, qa_pair in enumerate(qa_pairs):
        question = qa_pair['question']
        expected_answer = qa_pair['answer']
        expected_source = qa_pair.get('source', '')
        
        # Get prediction
        prediction = rag_system.query(question)
        predicted_answer = prediction['answer']
        predicted_sources = prediction.get('sources', [])
        
        results.append({
            'question': question,
            'expected_answer': expected_answer,
            'predicted_answer': predicted_answer,
            'expected_source': expected_source,
            'predicted_sources': predicted_sources
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(qa_pairs)} questions...")
    
    return results

if __name__ == "__main__":
    # Example usage
    rag_system = RAGSystem(use_rag=True)
    
    question = "What are the emergency contact numbers for disaster response?"
    result = rag_system.query(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")


