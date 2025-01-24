# Baddabing Menu Creator

A smart menu creation and event management system that leverages past catering events to create new menus and provide detailed event information.

## Overview

This application uses RAG (Retrieval Augmented Generation) to:
- Create custom menus based on previously successful events
- Retrieve detailed information about past events
- Provide accurate pricing breakdowns
- Manage catering event details

## Features

- **Smart Menu Creation**: Mix and match items from past events to create new, cohesive menus
- **Event Lookup**: Find detailed information about specific past events
- **Pricing Analysis**: Get detailed pricing breakdowns including per-person costs, staff charges, and additional fees
- **Document Processing**: Automatically process and index PDF menus from Google Drive
- **Interactive Chat Interface**: User-friendly Streamlit interface for queries and responses

## Prerequisites

- Python 3.9+
- Google Drive API credentials
- OpenAI API key


## Query Examples

- **Event Lookup**: "What was served at the Women's Entrepreneurial Opportunity Project event?"
- **Menu Creation**: "Create a lunch menu for 50 people with vegetarian options"
- **Pricing Info**: "What was the pricing breakdown for the Trial School event?"

## Features in Detail

### PDF Processing
- Extracts event details, menu items, and pricing information
- Processes structured pricing tables
- Identifies event metadata (dates, locations, contact info)

### Menu Creation
- Uses past menu items to create new combinations
- Maintains food pairing compatibility
- Provides pricing estimates based on historical data

### Event Information
- Stores complete event details
- Maintains pricing breakdowns
- Preserves setup notes and special instructions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## Acknowledgments

- Built with Streamlit
- Powered by OpenAI's GPT-4
- Uses LangChain for RAG implementation
- Google Drive integration via PyDrive2