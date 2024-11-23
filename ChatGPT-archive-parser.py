import unicodedata
import json
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine


# Helper function to load JSON data
def load_json(file_path):
    """Load a JSON file and return the data."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# Extract text parts from a message
def extract_message_parts(message: dict) -> list:
    """Extract text parts from a message.
    
    Args:
        message: Dictionary containing message data
        
    Returns:
        List of message parts
    """
    content = message.get("content", {})
    if content and content.get("content_type") == "text":
        return content.get("parts", [])
    return []


# Get the author name
def get_author_name(message: dict) -> str:
    """Get the standardized author name from message.
    
    Args:
        message: Dictionary containing message data
        
    Returns:
        String containing author name
    """
    author = message.get("author", {}).get("role", "")
    return {
        "assistant": "ChatGPT",
        "system": "Custom user info"
    }.get(author, author)


# Extract messages from a conversation
def get_conversation_messages(conversation: dict) -> list:
    """Extract all messages from a conversation.
    
    Args:
        conversation: Dictionary containing conversation data
        
    Returns:
        List of message dictionaries
    """
    messages = []
    current_node = conversation.get("current_node")
    mapping = conversation.get("mapping", {})
    
    while current_node:
        node = mapping.get(current_node, {})
        message = node.get("message")
        
        if not message:
            current_node = node.get("parent")
            continue
            
        parts = extract_message_parts(message)
        author = get_author_name(message)
        
        if (parts and parts[0] and 
            (author != "system" or 
             message.get("metadata", {}).get("is_user_system_message"))):
            
            create_time = message.get("create_time", 0)
            update_time = message.get("update_time", 0)
            
            messages.append({
                "author": author,
                "text": parts[0],
                "create_time": datetime.fromtimestamp(create_time),
                "update_time": datetime.fromtimestamp(update_time)
            })
            
        current_node = node.get("parent")
        
    return messages[::-1]


# Create a directory based on date
def create_directory(base_dir, date):
    directory_name = date.strftime("%Y_%m")
    directory_path = base_dir / directory_name
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


# Sanitize the title for valid file names
def sanitize_title(title):
    title = unicodedata.normalize("NFKC", title)
    title = re.sub(r'[<>:"/\\|?*\x00-\x1F\s]', '_', title)
    return title[:140]


# Create file name for conversation
def create_file_name(directory_path: Path, title: str, date: datetime) -> Path:
    """Create a sanitized file name for the conversation.
    
    Args:
        directory_path: Path object for target directory
        title: Conversation title
        date: Datetime object for file name
        
    Returns:
        Path object for the file
    """
    sanitized = sanitize_title(title)
    file_name = f"{date.strftime('%Y_%m_%d')}_{sanitized}.txt"
    return directory_path / file_name


# Write messages to text file
def write_messages_to_file(file_path, messages):
    with file_path.open("w", encoding="utf-8") as file:
        for message in messages:
            file.write(f"{message['author']}\n")
            file.write(f"{message['text']}\n")


# Extract data into DataFrames
def extract_conversations_to_df(conversations_data):
    conversation_records = []
    message_records = []

    for conversation in conversations_data:
        updated = conversation.get("update_time")
        if not updated:
            continue

        updated_date = datetime.fromtimestamp(updated) if updated else datetime.fromtimestamp(0)
        title = conversation.get("title", "Untitled")

        # Add conversation-level data
        conversation_records.append({
            "conversation_id": conversation.get("id", "Unknown"),
            "title": title,
            "create_time": datetime.fromtimestamp(conversation.get("create_time", 0)),
            "update_time": updated_date
        })

        # Extract messages
        messages = get_conversation_messages(conversation)
        for message in messages:
            message_records.append({
                "conversation_id": conversation.get("id", "Unknown"),
                "author": message['author'],
                "text": message['text'],
                "create_time": message['create_time'],
                "update_time": message['update_time']
            })

    conversations_df = pd.DataFrame(conversation_records)
    messages_df = pd.DataFrame(message_records)

    return conversations_df, messages_df


# Save DataFrames to a SQLite database
def save_to_database(
    conversations_df: pd.DataFrame,
    messages_df: pd.DataFrame, 
    db_uri: str = 'sqlite:///conversations.db'
) -> None:
    """Save DataFrames to SQLite database.
    
    Args:
        conversations_df: DataFrame with conversation data
        messages_df: DataFrame with message data
        db_uri: Database connection string
    """
    engine = create_engine(db_uri)
    with engine.connect() as conn:
        conversations_df.to_sql(
            "conversations",
            con=conn,
            if_exists='replace',
            index=False
        )
        messages_df.to_sql(
            "messages", 
            con=conn,
            if_exists='replace',
            index=False
        )


# Main function to process conversations and save data
def process_conversations(
    input_file: str,
    output_dir: str,
    db_uri: str = 'sqlite:///conversations.db'
) -> list:
    """Process conversations from JSON and save to files/database.
    
    Args:
        input_file: Path to input JSON file
        output_dir: Directory to save conversation files
        db_uri: Database connection string
        
    Returns:
        List of created directory/file information
    """
    try:
        input_path = Path(input_file)
        output_path = Path(output_dir)
        
        conversations_data = load_json(input_path)
        if not conversations_data:
            return []
            
        conversations_df, messages_df = extract_conversations_to_df(
            conversations_data
        )
        
        save_to_database(conversations_df, messages_df, db_uri)
        
        return save_conversation_files(
            conversations_df,
            messages_df, 
            output_path
        )
        
    except Exception as e:
        print(f"Error processing conversations: {e}")
        return []


# Example to run the function locally in Jupyter or as a script
def run_process(input_file='conversations.json', output_dir='output_directory'):
    """
    Run the process to extract data and save to files and database.
    """
    created_directories_info = process_conversations(input_file, output_dir)

    for info in created_directories_info:
        print(f"Created {info['file']} in directory {info['directory']}")


# You can call this function in a Jupyter Notebook or run it as a standalone script
# Example: 'conversations.json' should be replaced with the actual path
run_process('conversations.json', 'ChatGPT_Convos')
