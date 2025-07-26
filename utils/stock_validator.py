"""
Stock validation and import utilities
Validates stock symbols and handles data cleaning
"""

# List of known problematic/delisted stocks to exclude
DELISTED_STOCKS = {
    'ZOMATO', 'GMRINFRA', 'ADANITRANS', 'MINDTREE', 'L&TFH', 'MCDOWELL-N', 
    'JINDAL', 'IIFLWAM', 'PVR', 'POLICYBZR', 'PAYTM', 'NYKAA'
}

def validate_stock_symbol(symbol):
    """
    Validate if a stock symbol is valid and not delisted
    
    Args:
        symbol (str): Stock symbol to validate
        
    Returns:
        bool: True if valid, False if delisted or invalid
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Clean the symbol
    clean_symbol = symbol.strip().upper().replace('.NS', '')
    
    # Check if delisted
    if clean_symbol in DELISTED_STOCKS:
        return False
    
    # Basic symbol validation
    if len(clean_symbol) > 20 or len(clean_symbol) < 1:
        return False
    
    # Check for valid characters (allow alphanumeric, hyphen, ampersand)
    if not all(c.isalnum() or c in ['-', '&'] for c in clean_symbol):
        return False
    
    return True

def clean_stock_list(stock_list):
    """
    Clean a list of stock symbols by removing delisted/invalid ones
    
    Args:
        stock_list (list): List of stock symbols
        
    Returns:
        list: Cleaned list of valid symbols
    """
    cleaned = []
    for symbol in stock_list:
        if validate_stock_symbol(symbol):
            clean_symbol = symbol.strip().upper().replace('.NS', '')
            if clean_symbol not in cleaned:  # Remove duplicates
                cleaned.append(clean_symbol)
    
    return cleaned

def parse_imported_stocks(content, filename):
    """
    Parse imported stock file content and extract valid symbols
    
    Args:
        content (str): File content
        filename (str): Original filename
        
    Returns:
        dict: Dictionary with parsed stocks and metadata
    """
    result = {
        'stocks': [],
        'total_found': 0,
        'valid_stocks': 0,
        'invalid_stocks': [],
        'delisted_stocks': []
    }
    
    try:
        if filename.endswith('.csv'):
            # Handle CSV files
            lines = content.strip().split('\n')
            raw_stocks = []
            
            for line in lines:
                if line.strip():
                    parts = [p.strip().strip('"').upper() for p in line.split(',')]
                    # Try different columns that might contain stock symbols
                    for part in parts:
                        if part and len(part) <= 20:
                            raw_stocks.append(part)
        else:
            # Handle text files
            raw_stocks = [line.strip().upper() for line in content.split('\n') if line.strip()]
        
        result['total_found'] = len(raw_stocks)
        
        # Process each stock
        for stock in raw_stocks:
            clean_stock = stock.replace('.NS', '').strip()
            
            if clean_stock in DELISTED_STOCKS:
                result['delisted_stocks'].append(clean_stock)
            elif validate_stock_symbol(clean_stock):
                if clean_stock not in result['stocks']:  # Avoid duplicates
                    result['stocks'].append(clean_stock)
                    result['valid_stocks'] += 1
            else:
                result['invalid_stocks'].append(clean_stock)
        
        # Sort the final list
        result['stocks'] = sorted(result['stocks'])
        
    except Exception as e:
        result['error'] = str(e)
    
    return result