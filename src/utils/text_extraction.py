import re
from config.field_patterns import FIELDS

def extract_value(text, field):
    if field == "Email":
        match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        return (match.group(0), 1.0) if match else (None, 0.0)

    elif field == "Date of Journey" or field == "Date":
        patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',  # MM/DD/YYYY
            r'\b\d{2}-\d{2}-\d{4}\b',  # DD-MM-YYYY
            r'\b\d{1,2}\s+[A-Za-z]+\s*,\s*\d{4}\b'  # DD MMMM, YYYY
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return (match.group(0), 1.0)
        return (None, 0.0)

    elif field == "Mobile Number":
        match = re.search(r'\b(\+\d{1,3}[-.\s]?)?\d{10,14}\b', text)
        return (match.group(0), 1.0) if match else (None, 0.0)

    elif field == "Folio Number":
        match = re.search(r'\b[A-Za-z0-9]+\b', text)
        return (match.group(0), 0.8) if match else (None, 0.0)

    elif field == "PAN":
        match = re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', text)
        return (match.group(0), 1.0) if match else (None, 0.0)

    elif field == "Number of Units":
        match = re.search(r'\b\d+(\.\d+)?\b', text)
        return (match.group(0), 0.9) if match else (None, 0.0)

    elif field == "Tax Status":
        statuses = ["individual", "company", "huf", "nri", "trust"]
        for status in statuses:
            if status in text.lower():
                return (status.capitalize(), 0.9)
        return (text.split()[0], 0.5) if text else (None, 0.0)

    elif field == "Address":
        lines = text.split('\n')
        address_lines = []
        for line in lines:
            if any(re.search(pattern, line, re.IGNORECASE) for pattern in FIELDS.values()):
                break
            address_lines.append(line.strip())
        return (' '.join(address_lines), 0.7)

    elif field == "Bank Account Details":
        account_match = re.search(r'\b\d{9,18}\b', text)
        ifsc_match = re.search(r'\b[A-Z]{4}0[A-Z0-9]{6}\b', text)
        if account_match and ifsc_match:
            return (f"A/C: {account_match.group(0)}, IFSC: {ifsc_match.group(0)}", 1.0)
        elif account_match:
            return (f"A/C: {account_match.group(0)}", 0.8)
        return (text.strip(), 0.5)

    else:
        words = text.split()
        return (' '.join(words[:5]), 0.6)

def determine_label(field):
    question_fields = ["Scheme Name", "Folio Number", "Number of Units", "PAN", "Tax Status", 
                      "Mobile Number", "Email", "Address", "Date", "Bank Account Details", 
                      "Date of Journey"]
    if field in question_fields:
        return "question"
    elif field == "Signature":
        return "other"
    else:
        return "answer"
