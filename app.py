import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from config import OPENAI_API_KEY
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Baddabing Menu Creator",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RAGApplication:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        self.vector_store = FAISS.load_local(
            "faiss_index", 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name='gpt-4-0125-preview',
            openai_api_key=OPENAI_API_KEY,
            max_tokens=4000
        )
        
        # Create separate retrievers for different purposes
        self.event_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 8,
                "filter": {"document_type": "event_details"}
            }
        )
        
        self.food_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,  # Get more food items for better combinations
                "filter": {"document_type": "food_item"}
            }
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.event_retriever,  # Default to event retriever
            return_source_documents=True,
            verbose=True,
            max_tokens_limit=6000
        )

    def get_response(self, query, chat_history):
        """Get response from the LLM"""
        query_type = self._determine_query_type(query)
        
        if query_type == "menu_creation":
            return self._handle_menu_creation(query)
        elif query_type == "event_lookup":
            return self._handle_event_query(query)
        else:
            return self._handle_general_query(query)

    def _determine_query_type(self, query):
        """Determine the type of query"""
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in [
            "create", "make", "design", "suggest", "new menu",
            "what would you recommend", "can you prepare"
        ]):
            return "menu_creation"
        elif any(phrase in query_lower for phrase in [
            "what was", "tell me about", "details for", "information about",
            "pricing for", "menu for the", "what did we serve"
        ]):
            return "event_lookup"
        
        return "general"

    def _handle_menu_creation(self, query):
        """Handle menu creation requests using food item catalog"""
        context = """
        You are a creative menu designer with access to a catalog of food items. When creating menus:
        1. Use only food items that exist in the catalog
        2. Consider the menu section (appetizer, entree, etc.) when combining items
        3. Create cohesive combinations that work well together
        4. Reference which events the items were successfully used in
        5. Use the base pricing guidelines for new menus
        6. Explain why you chose each item and how they complement each other
        """
        
        # Get relevant food items
        food_docs = self.food_retriever.get_relevant_documents(query)
        
        # Create a focused menu creation prompt
        menu_prompt = f"""
        {context}
        
        Available Food Items:
        {self._format_food_items(food_docs)}
        
        Base Pricing Guidelines:
        - Drinks: $4 per person
        - Sandwich lunch: $20 per person
        - Hot lunch: $30 per person
        - Dinner: $30-$55 per person
        
        Query: {query}
        
        Create a menu using only the available items above. Include:
        1. Selected items with descriptions
        2. Why you chose each item
        3. Estimated pricing based on the guidelines
        4. Suggested variations or alternatives
        """
        
        with st.spinner("Creating custom menu..."):
            response = self.llm.predict(menu_prompt)
            return response

    def _format_food_items(self, food_docs):
        """Format food items for the menu creation prompt"""
        formatted_items = []
        for doc in food_docs:
            formatted_items.append(f"""
Item: {doc.metadata.get('item_name')}
Section: {doc.metadata.get('menu_section')}
Description: {doc.metadata.get('full_description')}
Previously used in: {doc.metadata.get('source_event')}
            """.strip())
        return "\n\n".join(formatted_items)

    def _handle_event_query(self, query):
        """Handle queries about specific events"""
        context = """
        You are a menu bot with access to detailed event records. When providing event information:
        1. Include the exact event name, date, and location
        2. List all menu items by section
        3. Provide the complete pricing breakdown
        4. Include setup notes and contact information
        Be specific and use exact details from the event records.
        """
        
        result = self.qa_chain({
            "question": f"{context}\n\nFind complete details for this event: {query}",
            "chat_history": []
        })
        return result["answer"]

    def _handle_general_query(self, query):
        """Handle general queries"""
        context = "Menu creator bot. Use existing items/pricing only."
        result = self.qa_chain({
            "question": f"{context}\n\nQuery: {query}",
            "chat_history": []
        })
        return result["answer"]

def initialize_session_state():
    """Initialize session state variables"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

def main():
    initialize_session_state()
    
    st.title("Baddabing Menu Creator")
    
    # Add description
    st.markdown("""
    Welcome to the Menu Creator! This tool helps you create custom menus using existing 
    items from our catering service. All suggestions will be based on our past menu 
    items and pricing.
    """)
    
    try:
        # Initialize RAG application
        rag_app = RAGApplication()
        
        # Display only the last 5 exchanges in the UI
        display_history = st.session_state.chat_history[-5:] if len(st.session_state.chat_history) > 5 else st.session_state.chat_history
        
        # Display chat history
        for question, answer in display_history:
            st.write("🤔 You:", question)
            st.write("🤖 Menu Creator:", answer)
            st.write("---")
        
        # Create a form for input
        with st.form(key="chat_form", clear_on_submit=True):
            user_question = st.text_area(
                "What kind of menu would you like to create?",
                placeholder="Example: Create a lunch menu for 20 people...",
                height=150,
                key="user_question"
            )
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_question:
                # Get new response
                response = rag_app.get_response(user_question, [])  # Pass empty chat history
                
                # Update session state with new exchange
                st.session_state.chat_history.append((user_question, response))
                
                # Limit stored history to last 10 exchanges
                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = st.session_state.chat_history[-10:]
                
                st.rerun()
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if os.environ.get('DEBUG'):
            st.exception(e)

if __name__ == "__main__":
    main() 