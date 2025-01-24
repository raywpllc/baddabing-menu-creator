from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import PyPDF2
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from config import GOOGLE_DRIVE_CREDENTIALS, DRIVE_FOLDER_ID, OPENAI_API_KEY
from menu_base_pricing import BASE_PRICING
import os
import re
import json

class PDFProcessor:
    def __init__(self):
        self.gauth = GoogleAuth()
        # Load settings
        self.gauth.settings['client_config_file'] = 'client_secrets.json'
        self.gauth.settings['save_credentials'] = True
        self.gauth.settings['save_credentials_backend'] = 'file'
        self.gauth.settings['save_credentials_file'] = 'credentials.json'
        self.gauth.settings['get_refresh_token'] = True
        self.gauth.settings['oauth_scope'] = [
            'https://www.googleapis.com/auth/drive',
            'https://www.googleapis.com/auth/drive.metadata.readonly'
        ]
        
        try:
            # Try to load saved client credentials
            self.gauth.LoadCredentialsFile("credentials.json")
            
            if self.gauth.credentials is None:
                # Authenticate if nothing is stored
                try:
                    self.gauth.LocalWebserverAuth(port_numbers=[8080])
                except:
                    # If 8080 fails, try alternate ports
                    try:
                        self.gauth.LocalWebserverAuth(port_numbers=[8081])
                    except:
                        self.gauth.LocalWebserverAuth(port_numbers=[8090])
            elif self.gauth.access_token_expired:
                # Refresh them if expired
                self.gauth.Refresh()
            else:
                # Initialize the saved creds
                self.gauth.Authorize()
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            # Try one last time with different port
            try:
                self.gauth.LocalWebserverAuth(port_numbers=[8088])
            except Exception as auth_error:
                print(f"Final authentication attempt failed: {str(auth_error)}")
                raise Exception("Could not authenticate with Google Drive")
        
        # Save the current credentials
        self.gauth.SaveCredentialsFile("credentials.json")
        self.drive = GoogleDrive(self.gauth)
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def extract_pricing_details(self, text):
        """Extract detailed pricing breakdown with line item associations"""
        pricing_details = {
            'per_person_charges': [],  # Items charged per guest
            'flat_charges': [],        # Fixed charges
            'staff_charges': [],       # Staff-related charges
            'additional_charges': [],  # Additional services
            'summary': {              # Total calculations
                'subtotal': None,
                'service_fee': None,
                'delivery_setup': None,
                'tax': None,
                'tax_rate': None,
                'grand_total': None
            }
        }
        
        # Find the pricing section
        pricing_section = None
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if re.match(r'^\s*pricing\s*$', line, re.I):
                # Capture all lines until we hit a blank line or new section
                pricing_lines = []
                j = i + 1
                while j < len(lines) and lines[j].strip() and not lines[j].strip().lower().startswith(('menu', 'contact')):
                    pricing_lines.append(lines[j].strip())
                    j += 1
                pricing_section = '\n'.join(pricing_lines)
                break
        
        if not pricing_section:
            return pricing_details
        
        # Patterns for different pricing formats
        patterns = {
            'per_person': r'(?P<item>.*?)\s*at\s*\$(?P<price>[\d,.]+)\s*per\s*guest\s*x\s*(?P<guests>\d+)\s*guests?\s*=\s*\$(?P<total>[\d,.]+)',
            'staff': r'(?P<role>.*?)\s*at\s*\$(?P<rate>[\d,.]+)\s*x\s*(?P<count>\d+)?\s*=\s*\$(?P<total>[\d,.]+)',
            'tax': r'(?P<rate>[\d.]+)\s*%\s*tax\s*=\s*\$(?P<amount>[\d,.]+)',
            'service': r'service\s*fee\s*=\s*\$(?P<amount>[\d,.]+)',
            'delivery': r'delivery\s*(?:&|and)\s*set-?up\s*fee\s*=\s*\$(?P<amount>[\d,.]+)',
            'total': r'grand\s*total\s*=\s*\$(?P<amount>[\d,.]+)',
            'flat_rate': r'(?P<item>.*?)\s*=\s*\$(?P<amount>[\d,.]+)',
            'tbd': r'(?P<item>.*?)\s*=\s*(?:t\.b\.d\.|TBD)',
        }
        
        # Process each line of the pricing section
        for line in pricing_section.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Try to match each pattern
            matched = False
            
            # Check for per-person charges first
            match = re.match(patterns['per_person'], line, re.I)
            if match:
                pricing_details['per_person_charges'].append({
                    'item': match.group('item').strip(),
                    'price_per_person': float(match.group('price').replace(',', '')),
                    'guest_count': int(match.group('guests')),
                    'total': float(match.group('total').replace(',', '')),
                    'line_item': line.strip()
                })
                matched = True
                continue
            
            # Check for staff charges
            match = re.match(patterns['staff'], line, re.I)
            if match and 'guest' not in line.lower():
                pricing_details['staff_charges'].append({
                    'role': match.group('role').strip(),
                    'rate': float(match.group('rate').replace(',', '')),
                    'count': int(match.group('count')) if match.group('count') else 1,
                    'total': float(match.group('total').replace(',', '')),
                    'line_item': line.strip()
                })
                matched = True
                continue
            
            # Check for tax
            match = re.match(patterns['tax'], line, re.I)
            if match:
                pricing_details['summary']['tax_rate'] = float(match.group('rate'))
                pricing_details['summary']['tax'] = float(match.group('amount').replace(',', ''))
                matched = True
                continue
            
            # Check for service fee
            match = re.match(patterns['service'], line, re.I)
            if match:
                pricing_details['summary']['service_fee'] = float(match.group('amount').replace(',', ''))
                matched = True
                continue
            
            # Check for delivery fee
            match = re.match(patterns['delivery'], line, re.I)
            if match:
                pricing_details['summary']['delivery_setup'] = float(match.group('amount').replace(',', ''))
                matched = True
                continue
            
            # Check for grand total
            match = re.match(patterns['total'], line, re.I)
            if match:
                pricing_details['summary']['grand_total'] = float(match.group('amount').replace(',', ''))
                matched = True
                continue
            
            # Check for TBD items
            match = re.match(patterns['tbd'], line, re.I)
            if match:
                pricing_details['additional_charges'].append({
                    'item': match.group('item').strip(),
                    'amount': 'TBD',
                    'line_item': line.strip()
                })
                matched = True
                continue
            
            # Check for flat rate items if no other patterns matched
            if not matched:
                match = re.match(patterns['flat_rate'], line, re.I)
                if match:
                    item = match.group('item').strip().lower()
                    if not any(keyword in item for keyword in ['total', 'sub-total', 'service fee', 'tax']):
                        pricing_details['flat_charges'].append({
                            'item': match.group('item').strip(),
                            'amount': float(match.group('amount').replace(',', '')),
                            'line_item': line.strip()
                        })
        
        return pricing_details

    def extract_event_details(self, text, filename):
        """Extract comprehensive event details"""
        # Enhanced patterns for event metadata
        patterns = {
            'price': [
                r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:per person|pp|p/p))?',
                r'(?:price|cost|total):\s*\$[\d,]+(?:\.\d{2})?',
                r'(?:per person|pp|p/p):\s*\$[\d,]+(?:\.\d{2})?'
            ],
            'date': [
                r'(?:date:|on:?)\s*([\w\s,.&]+\d{2,4})',
                r'(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
                r'([A-Z][a-z]+ \d{1,2}(?:st|nd|rd|th)?,? \d{4})'
            ],
            'time': [
                r'(?:time[s]?:|at:?)\s*([^\n]+)',
                r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)(?:\s*-\s*\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))?)'
            ],
            'guest_count': [
                r'(?:guests?|people|attendees|count):\s*(\d+)',
                r'(?:for|serving)\s+(\d+)\s+(?:people|guests|attendees)',
                r'guests?:\s*([^\n]+)'  # Capture varying guest counts
            ],
            'location': [
                r'(?:location|venue|place):\s*([^\n]+)'
            ],
            'setup_notes': [
                r'(?:setup|set up|setup notes):\s*([^\n]+(?:\n(?!\w+:)[^\n]+)*)'
            ],
            'invoice_no': [
                r'(?:invoice\s*(?:no|number|#)?:?\s*)([A-Z0-9]+)'
            ],
            'contact': [
                r'(?:contact|contact person):\s*([^\n]+)'
            ],
            'email': [
                r'(?:email|e-mail):\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
            ],
            'phone': [
                r'(?:phone|cell|tel):\s*([0-9.()-]+)',
                r'(?:phone|cell|tel)[^:]*:\s*([^\n]+)'  # Capture phone with description
            ]
        }
        
        # Add specific pattern for event name
        event_patterns = [
            r'^(?:event|function|occasion):\s*(.+)$',
            r'^(?:event|function|occasion)\s+name:\s*(.+)$',
            r'(?:catering|menu)\s+for:\s*(.+)$'
        ]
        
        # Initialize details dictionary
        details = {
            "event_name": "",
            "date": None,
            "time": None,
            "guest_count": None,
            "location": None,
            "setup_notes": None,
            "invoice_no": None,
            "contact": None,
            "email": None,
            "phone": None,
            "prices": [],
            "menu_items": [],
            "full_text": text
        }
        
        # Extract event name - enhanced method
        lines = text.split('\n')
        event_name_found = False
        
        # First try explicit event patterns
        for line in lines[:15]:  # Look in first 15 lines
            line = line.strip()
            for pattern in event_patterns:
                match = re.match(pattern, line, re.I)
                if match:
                    details['event_name'] = match.group(1).strip()
                    event_name_found = True
                    break
            if event_name_found:
                break
        
        # If no explicit event name found, look for a suitable header
        if not event_name_found:
            for line in lines[:10]:
                line = line.strip()
                # Look for lines that might be event names
                if (len(line) > 10 and 
                    not any(p in line.lower() for p in ['date:', 'time:', 'price:', 'location:', 'menu:', 'contact:']) and
                    not line.startswith('$') and
                    not re.match(r'^\d+', line)):
                    details['event_name'] = line
                    break
        
        # If still no event name, use filename
        if not details['event_name']:
            details['event_name'] = filename.replace('.pdf', '').replace('_', ' ').title()
        
        # Create searchable variations of the event name
        details['event_name_variations'] = [
            details['event_name'],
            details['event_name'].lower(),
            re.sub(r'[^\w\s]', '', details['event_name']).lower(),  # Remove punctuation
            ' '.join(word for word in details['event_name'].split() if len(word) > 2)  # Key words only
        ]
        
        # Extract all metadata using patterns
        for field, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if field == 'price':
                        details['prices'].append(match.group())
                    elif field == 'setup_notes':
                        # Capture multiline setup notes
                        setup_text = match.group(1)
                        details['setup_notes'] = setup_text.strip()
                    else:
                        if not details[field]:  # Only capture first match for non-price fields
                            details[field] = match.group(1).strip()
        
        # Extract menu items
        menu_section = False
        current_section = ""
        menu_sections = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect menu section headers
            if re.search(r'menu|breakfast|lunch|dinner|appetizers|entrees|desserts|beverages', line, re.I):
                menu_section = True
                current_section = line
                menu_sections[current_section] = []
            elif menu_section and not line.startswith(('$', 'Price', 'Total', 'Contact', 'Phone', 'Email')):
                if current_section in menu_sections:
                    menu_sections[current_section].append(line)
        
        # Format menu items with sections
        details['menu_items'] = menu_sections
        
        # Add pricing details extraction
        pricing_details = self.extract_pricing_details(text)
        details['pricing_breakdown'] = pricing_details
        
        return details

    def create_documents(self, event_details):
        """Create documents with enhanced metadata"""
        documents = []
        
        # Create main event document
        main_doc = Document(
            page_content=event_details["full_text"],
            metadata={
                "event_name": event_details["event_name"],
                "date": event_details["date"],
                "time": event_details["time"],
                "guest_count": event_details["guest_count"],
                "location": event_details["location"],
                "setup_notes": event_details["setup_notes"],
                "invoice_no": event_details["invoice_no"],
                "contact": event_details["contact"],
                "email": event_details["email"],
                "phone": event_details["phone"],
                "prices": event_details["prices"],
                "menu_items": event_details["menu_items"],
                "document_type": "event_menu"
            }
        )
        documents.append(main_doc)
        
        # Create specific event details document
        details_text = f"""
Event: {event_details['event_name']}
Date: {event_details['date'] or 'Not specified'}
Time: {event_details['time'] or 'Not specified'}
Guest Count: {event_details['guest_count'] or 'Not specified'}
Location: {event_details['location'] or 'Not specified'}
Invoice: {event_details['invoice_no'] or 'Not specified'}
Contact: {event_details['contact'] or 'Not specified'}
Email: {event_details['email'] or 'Not specified'}
Phone: {event_details['phone'] or 'Not specified'}

Setup Notes:
{event_details['setup_notes'] or 'No setup notes provided'}

Pricing Information:
{chr(10).join(f"- {price}" for price in event_details['prices'])}

Menu Items:
"""
        
        # Add menu sections
        for section, items in event_details['menu_items'].items():
            details_text += f"\n{section}:\n"
            for item in items:
                details_text += f"- {item}\n"
        
        # Update details_text to include pricing breakdown
        details_text += "\nDetailed Pricing Breakdown:\n"
        
        if event_details['pricing_breakdown']['per_person_charges']:
            details_text += "\nPer Person Charges:\n"
            for charge in event_details['pricing_breakdown']['per_person_charges']:
                details_text += f"- {charge['line_item']}\n"
        
        if event_details['pricing_breakdown']['staff_charges']:
            details_text += "\nStaff Charges:\n"
            for charge in event_details['pricing_breakdown']['staff_charges']:
                details_text += f"- {charge['line_item']}\n"
        
        if event_details['pricing_breakdown']['flat_charges']:
            details_text += "\nAdditional Charges:\n"
            for charge in event_details['pricing_breakdown']['flat_charges']:
                details_text += f"- {charge['line_item']}\n"
        
        if event_details['pricing_breakdown']['additional_charges']:
            details_text += "\nTBD Charges:\n"
            for charge in event_details['pricing_breakdown']['additional_charges']:
                details_text += f"- {charge['line_item']}\n"
        
        summary = event_details['pricing_breakdown']['summary']
        if any(summary.values()):
            details_text += "\nSummary:\n"
            if summary['subtotal']:
                details_text += f"Subtotal: ${summary['subtotal']:.2f}\n"
            if summary['service_fee']:
                details_text += f"Service Fee: ${summary['service_fee']:.2f}\n"
            if summary['delivery_setup']:
                details_text += f"Delivery & Setup: ${summary['delivery_setup']:.2f}\n"
            if summary['tax']:
                details_text += f"Tax ({summary['tax_rate']}%): ${summary['tax']:.2f}\n"
            if summary['grand_total']:
                details_text += f"Grand Total: ${summary['grand_total']:.2f}\n"
        
        details_doc = Document(
            page_content=details_text,
            metadata={
                "event_name": event_details["event_name"],
                "document_type": "event_details",
                **{k: v for k, v in event_details.items() if k != 'full_text'}
            }
        )
        documents.append(details_doc)
        
        return documents

    def extract_text_from_pdf(self, file):
        """Extract text content from a PDF file"""
        try:
            # Download the file content
            print(f"Downloading content for {file['title']}...")
            file.GetContentFile('temp.pdf')
            
            # Read the downloaded file
            print(f"Reading PDF content for {file['title']}...")
            with open('temp.pdf', 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                total_pages = len(pdf_reader.pages)
                print(f"Found {total_pages} pages in {file['title']}")
                
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += page_text + "\n"
                        print(f"Extracted {len(page_text)} characters from page {i+1}")
                    except Exception as e:
                        print(f"Error extracting text from page {i+1}: {str(e)}")
            
            # Clean up the temporary file
            os.remove('temp.pdf')
            return text
            
        except Exception as e:
            print(f"Error processing PDF {file['title']}: {str(e)}")
            return ""

    def process_all_pdfs(self):
        """Process PDFs and create enhanced embeddings"""
        self.authenticate_google_drive()
        pdf_files = self.get_pdf_files()
        
        print(f"Found {len(pdf_files)} PDF files in the specified folder")
        if len(pdf_files) == 0:
            raise Exception(f"No PDF files found in folder ID: {DRIVE_FOLDER_ID}")
        
        all_documents = []
        event_summaries = []
        
        for pdf_file in pdf_files:
            print(f"Processing file: {pdf_file['title']}")
            try:
                text = self.extract_text_from_pdf(pdf_file)
                if text.strip():
                    event_details = self.extract_event_details(text, pdf_file['title'])
                    documents = self.create_documents(event_details)
                    all_documents.extend(documents)
                    
                    # Create summary for this event
                    summary = {
                        "event_name": event_details["event_name"],
                        "date": event_details["date"],
                        "prices": event_details["prices"],
                        "guest_count": event_details["guest_count"],
                        "menu_items": event_details["menu_items"]
                    }
                    event_summaries.append(summary)
                    
                    print(f"Created {len(documents)} documents for {pdf_file['title']}")
                    print(f"Found prices: {event_details['prices']}")
            except Exception as e:
                print(f"Error processing {pdf_file['title']}: {str(e)}")
        
        if not all_documents:
            raise Exception("No documents were created from the PDF files")
        
        # Save event summaries
        with open('event_summaries.json', 'w') as f:
            json.dump(event_summaries, f, indent=2)
        
        # Add base pricing document
        base_pricing_doc = Document(
            page_content=f"Base Menu Pricing:\n{json.dumps(BASE_PRICING, indent=2)}",
            metadata={"document_type": "base_pricing"}
        )
        all_documents.append(base_pricing_doc)
        
        print(f"Creating vector store with {len(all_documents)} documents")
        vector_store = FAISS.from_documents(all_documents, self.embeddings)
        vector_store.save_local("faiss_index")
        return vector_store

    def get_pdf_files(self):
        """Get all PDF files from the specified Google Drive folder"""
        query = f"'{DRIVE_FOLDER_ID}' in parents and mimeType='application/pdf'"
        file_list = self.drive.ListFile({'q': query}).GetList()
        return file_list

    def authenticate_google_drive(self):
        """Authenticate with Google Drive"""
        if self.gauth.credentials is None:
            self.gauth.LocalWebserverAuth()
        elif self.gauth.access_token_expired:
            self.gauth.Refresh()
        else:
            self.gauth.Authorize()
        self.gauth.SaveCredentialsFile("credentials.json") 