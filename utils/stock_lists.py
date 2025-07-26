# Indian Stock Market - Major indices and sector-wise stock lists

# Updated NSE NIFTY 50 - Current as of 2024
NIFTY_50 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'BHARTIARTL', 'ICICIBANK', 'INFY', 'SBIN', 'LICI',
    'ITC', 'HINDUNILVR', 'LT', 'HCLTECH', 'MARUTI', 'SUNPHARMA', 'BAJFINANCE', 'ONGC',
    'KOTAKBANK', 'TITAN', 'ASIANPAINT', 'NESTLEIND', 'ULTRACEMCO', 'WIPRO', 'DMART',
    'BAJAJFINSV', 'M&M', 'NTPC', 'ADANIENT', 'COALINDIA', 'AXISBANK', 'POWERGRID',
    'JSWSTEEL', 'GRASIM', 'HDFCLIFE', 'TATAMOTORS', 'TECHM', 'INDUSINDBK', 'HINDALCO',
    'BRITANNIA', 'CIPLA', 'BAJAJ-AUTO', 'SBILIFE', 'BPCL', 'DRREDDY', 'APOLLOHOSP',
    'DIVISLAB', 'TATACONSUM', 'HEROMOTOCO', 'EICHERMOT', 'TATASTEEL', 'ADANIPORTS'
]

NIFTY_100 = NIFTY_50 + [
    'GODREJCP', 'PIDILITIND', 'COLPAL', 'BERGEPAINT', 'MARICO', 'DABUR', 'PAGEIND',
    'VBL', 'UBL', 'AMBUJACEM', 'SHREECEM', 'ACC', 'JKCEMENT', 'RAMCOCEM',
    'SAIL', 'NMDC', 'VEDL', 'JINDALSTEL', 'TATAPOWER', 'ADANIGREEN',
    'IOC', 'GAIL', 'PETRONET', 'ATGL', 'ICICIPRULI', 'BAJAJHLDNG', 'PFC', 'RECLTD', 'IRFC',
    'BANKBARODA', 'CANBK', 'PNB', 'UNIONBANK', 'IDFCFIRSTB', 'FEDERALBNK', 'RBLBANK'
]

NIFTY_200 = NIFTY_100 + [
    'ABCAPITAL', 'ABFRL', 'ADANIPOWER', 'AIAENG', 'ALKEM', 'APLLTD', 'AUROPHARMA',
    'BALKRISIND', 'BANDHANBNK', 'BATAINDIA', 'BEL', 'BHARATFORG', 'BHEL', 'BIOCON',
    'BOSCHLTD', 'CHOLAFIN', 'CONCOR', 'COROMANDEL', 'CUMMINSIND', 'DEEPAKNTR',
    'DELTACORP', 'DLF', 'ESCORTS', 'EXIDEIND', 'FORTIS', 'GLENMARK',
    'GODREJPROP', 'GRANULES', 'GUJGASLTD', 'HAL', 'HAVELLS', 'HINDPETRO',
    'IDEA', 'IGL', 'INDIGO', 'INDUSTOWER', 'IRCTC', 'JUBLFOOD', 'LICHSGFIN', 
    'LUPIN', 'MANAPPURAM', 'MFSL', 'MPHASIS', 'MRF', 'MUTHOOTFIN', 'NATIONALUM', 
    'NAUKRI', 'NAVINFLUOR', 'OBEROIRLTY', 'OFSS', 'OIL', 'PERSISTENT', 'PIIND',
    'POLYCAB', 'SBICARD', 'SIEMENS', 'SRF', 'SUPREMEIND', 'TORNTPHARM', 'TRENT', 
    'TVSMOTOR', 'UPL', 'VOLTAS', 'WHIRLPOOL', 'YESBANK', 'ZEEL'
]

# Sector-wise classification
SECTOR_STOCKS = {
    'IT': [
        'TCS', 'INFOSYS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'MINDTREE', 'MPHASIS', 'LTI',
        'OFSS', 'PERSISTENT', 'COFORGE', 'LTTS', 'KPITTECH', 'ZENSAR', 'NIIT',
        'RAMSARUP', 'SONATSOFTW', 'SASKEN', 'NELCO', 'CYIENT'
    ],
    
    'Banking & Financial Services': [
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK',
        'BANKBARODA', 'PNB', 'CANBK', 'UNIONBANK', 'IDFCFIRSTB', 'FEDERALBNK',
        'RBLBANK', 'YESBANK', 'CITYUNION', 'DCBBANK', 'SOUTHBANK', 'INDIANB',
        'BAJFINANCE', 'BAJAJFINSV', 'HDFCLIFE', 'SBILIFE', 'ICICIPRULI', 'LICHSGFIN',
        'BAJAJHLDNG', 'PFC', 'RECLTD', 'IRFC', 'ABCAPITAL', 'CHOLAFIN', 'MANAPPURAM',
        'MUTHOOTFIN', 'MFSL', 'SBICARD'
    ],
    
    'Pharmaceuticals': [
        'SUNPHARMA', 'CIPLA', 'DRREDDY', 'DIVISLAB', 'LUPIN', 'AUROBINDO', 'BIOCON',
        'TORNTPHARM', 'CADILAHC', 'ALKEM', 'GLENMARK', 'ABBOTINDIA', 'AUROPHARMA',
        'LALPATHLAB', 'METROPOLIS', 'THYROCARE', 'APOLLOHOSP', 'FORTIS'
    ],
    
    'FMCG': [
        'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'TATACONSUM', 'GODREJCP',
        'DABUR', 'MARICO', 'COLPAL', 'VBL', 'EMAMILTD', 'JYOTHYLAB', 'RADICO',
        'KRBL', 'TASTYBITE', 'HATSUN', 'PRATAAP', 'PIDILITIND', 'PAGEIND', 'UBL'
    ],
    
    'Automotive': [
        'MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'EICHERMOT',
        'TVSMOTOR', 'ASHOKLEY', 'BHARATFORG', 'BALKRISIND', 'MRF', 'APOLLOTYRE',
        'CEAT', 'ESCORTS', 'FORCE', 'MAHSCOOTER', 'EXIDEIND', 'BOSCHLTD'
    ],
    
    'Metals & Mining': [
        'JSWSTEEL', 'TATASTEEL', 'HINDALCO', 'COALINDIA', 'SAIL', 'NMDC', 'VEDL',
        'JINDALSTEL', 'NATIONALUM', 'MOIL', 'APL', 'WELCORP', 'JSPL', 'RATNAMANI'
    ],
    
    'Oil, Gas & Energy': [
        'RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'BPCL', 'IOC', 'HPCL', 'GAIL',
        'PETRONET', 'OIL', 'HINDPETRO', 'TATAPOWER', 'ADANIGREEN', 'ADANITRANS',
        'ADANIPOWER', 'SJVN', 'NHPC', 'PFC', 'RECLTD', 'ADANIENT', 'ADANIPORTS'
    ],
    
    'Cement & Construction': [
        'ULTRACEMCO', 'GRASIM', 'AMBUJACEM', 'SHREECEM', 'ACC', 'JKCEMENT',
        'RAMCOCEM', 'HEIDELBERG', 'PRISM', 'ORIENT', 'DALMIACEMT', 'INDIACEM',
        'LT', 'DLF', 'GODREJPROP', 'OBEROIRLTY'
    ],
    
    'Telecom': [
        'BHARTIARTL', 'IDEA', 'INDUSTOWER', 'HFCL', 'GTPL', 'RCOM', 'RAILTEL',
        'ROUTE', 'TATACOMM', 'TEJAS'
    ],
    
    'Capital Goods': [
        'BEL', 'BHEL', 'HAL', 'SIEMENS', 'ABB', 'THERMAX', 'CUMMINSIND',
        'VOLTAS', 'KECL', 'GREAVESCOT', 'TIINDIA', 'KIRLOSIND', 'VAIBHAVGBL'
    ],
    
    'Consumer Discretionary': [
        'TITAN', 'DMART', 'VBL', 'TRENT', 'NYKAA', 'ZOMATO', 'PAYTM', 'BATAINDIA',
        'WHIRLPOOL', 'HAVELLS', 'DIXON', 'AMBER', 'CROMPTON', 'SYMPHONY'
    ],
    
    'Media & Entertainment': [
        'ZEEL', 'SUNTV', 'PVRINOX', 'INOXLEISUR', 'TIPS', 'NAVNETEDUL', 'EROS'
    ],
    
    'Textiles': [
        'ABFRL', 'RAYMOND', 'ARVIND', 'VARDHMAN', 'WELSPUNIND', 'TRIDENT', 'CENTEXT'
    ],
    
    'Chemicals': [
        'SRF', 'UPL', 'PIDILITIND', 'AARTI', 'COROMANDEL', 'DEEPAKNTR', 'GHCL',
        'BALRAMCHIN', 'TATACHEM', 'GNFC', 'CHAMBLFERT', 'NAVINFLUOR'
    ]
}

# Market cap based classification
LARGE_CAP = NIFTY_50

MID_CAP = [
    'ABCAPITAL', 'ABFRL', 'ACC', 'ADANIPOWER', 'AIAENG', 'ALKEM', 'AMBUJACEM',
    'APOLLOHOSP', 'AUROPHARMA', 'BANDHANBNK', 'BANKBARODA', 'BATAINDIA', 'BEL',
    'BHARATFORG', 'BHEL', 'BIOCON', 'BOSCHLTD', 'CANBK', 'CHOLAFIN', 'CIPLA',
    'COALINDIA', 'CONCOR', 'COROMANDEL', 'CUMMINSIND', 'DABUR', 'DEEPAKNTR',
    'DELTACORP', 'DLF', 'DRREDDY', 'EICHERMOT', 'ESCORTS', 'EXIDEIND', 'FEDERALBNK',
    'FORTIS', 'GLENMARK', 'GMRINFRA', 'GODREJCP', 'GODREJPROP', 'GRANULES',
    'GUJGASLTD', 'HAL', 'HAVELLS', 'HCLTECH', 'HINDPETRO', 'HINDUNILVR',
    'IBULHSGFIN', 'IDEA', 'IDFCFIRSTB', 'IGL', 'INDIGO', 'INDUSTOWER',
    'IRCTC', 'ITC', 'JINDALSTEL', 'JUBLFOOD', 'L&TFH', 'LICHSGFIN', 'LUPIN',
    'MANAPPURAM', 'MARICO', 'MFSL', 'MINDTREE', 'MPHASIS', 'MRF', 'MUTHOOTFIN',
    'NATIONALUM', 'NAUKRI', 'NAVINFLUOR', 'NESTLEIND', 'OBEROIRLTY', 'OFSS',
    'OIL', 'PERSISTENT', 'PETRONET', 'PIIND', 'PNB', 'POLICYBZR', 'POLYCAB',
    'PVR', 'RBLBANK', 'SAIL', 'SBICARD', 'SHREECEM', 'SIEMENS', 'SRF',
    'STAR', 'SUPREMEIND', 'TATASTEEL', 'TORNTPHARM', 'TRENT', 'TVSMOTOR',
    'UPL', 'VEDL', 'VOLTAS', 'WHIRLPOOL', 'YESBANK', 'ZEEL', 'ZOMATO'
]

SMALL_CAP = [
    'AAVAS', 'AFFLE', 'ANANTRAJ', 'ANGEL', 'APARINDS', 'ASHOKA', 'ASTERDM',
    'ASTRAZEN', 'ATUL', 'AVANTI', 'AXISBANK', 'BAJAJHLDNG', 'BALRAMCHIN',
    'BEML', 'BERGEPAINT', 'BSOFT', 'CANFINHOME', 'CAPLIPOINT', 'CARBORUNIV',
    'CARERATING', 'CDSL', 'CENTRALBK', 'CENTURYPLY', 'CGPOWER', 'CHAMBERS',
    'CHEMCON', 'CHEMPLASTS', 'CLEAN', 'CMSINFO', 'COCHINSHIP', 'COLPAL',
    'COROMANDEL', 'CREDITACC', 'CROMPTON', 'CSBBANK', 'DCBBANK', 'DEEPAKFERT',
    'DELTACORP', 'DEVYANI', 'DHANI', 'DHANUKA', 'DIXON', 'DMART', 'EASEMYTRIP',
    'EDELWEISS', 'EMIL', 'EQUITAS', 'ESABINDIA', 'FINEORG', 'FINPIPE',
    'FORTIS', 'FSL', 'GICRE', 'GILLETTE', 'GLAND', 'GNFC', 'GPPL',
    'GRAPHITE', 'GREENLAM', 'GRINDWELL', 'GULFOILLUB', 'HAPPSTMNDS', 'HATHWAY',
    'HFCL', 'HINDCOPPER', 'HONAUT', 'HUDCO', 'IBREALEST', 'ICICIPRULI',
    'IIFL', 'INDHOTEL', 'INDIAMART', 'INDIANHUME', 'INDOCO', 'INFIBEAM',
    'INOXLEISUR', 'INTELLECT', 'IRCON', 'ISEC', 'JCHAC', 'JINDALSAW',
    'JKPAPER', 'JMFINANCIL', 'JSL', 'JUSTDIAL', 'JYOTHYLAB', 'KALPATPOWR',
    'KANSAINER', 'KARURVYSYA', 'KEI', 'KIMS', 'KPRMILL', 'KRBL', 'KSCL',
    'LAXMIMACH', 'LEMONTREE', 'LGBBROSLTD', 'MAHLOG', 'MANAPPURAM', 'MAPMYINDIA',
    'MASTEK', 'MCDOWELL-N', 'MCX', 'MEDPLUS', 'MGMRESORTS', 'MINDACORP',
    'MOIL', 'MOTHERSON', 'MRPL', 'MTARTECH', 'NATCOPHARM', 'NESCO',
    'NETWORK18', 'NEWGEN', 'NIITLTD', 'NLCINDIA', 'NOCIL', 'NUVOCO',
    'ORIENTREF', 'PANAMAPET', 'PARAGMILK', 'PCJEWELLER', 'PGHL', 'PNBHOUSING',
    'POLYMED', 'POWERMECH', 'PRSMJOHNSN', 'QUESS', 'RADICO', 'RAILTEL',
    'RAIN', 'RAJESHEXPO', 'RAMCOCEM', 'RATNAMANI', 'RELAXO', 'RHIM',
    'RITES', 'RVNL', 'SADBHAV', 'SANDUMA', 'SANOFI', 'SCHAEFFLER',
    'SHILPAMED', 'SHOPERSTOP', 'SHYAMMETL', 'SOLARA', 'SONATSOFTW', 'SPANDANA',
    'SPARC', 'SRTRANSFIN', 'STARCEMENT', 'STLTECH', 'SUBEXLTD', 'SUDARSCHEM',
    'SUPRAJIT', 'SYMPHONY', 'TAKE', 'TATACHEM', 'TATACOMM', 'TATAINVEST',
    'TCNSBRANDS', 'TEAM', 'TECHM', 'THYROCARE', 'TIINDIA', 'TI', 'TRIDENT',
    'TTKPRESTIG', 'TV18BRDCST', 'TVSHLTD', 'UCOBANK', 'UJJIVAN', 'UNOMINDA',
    'USHAMART', 'UTIAMC', 'VAIBHAVGBL', 'VAKRANGEE', 'VARROC', 'VGUARD',
    'VINATIORGA', 'VIPIND', 'VSTIND', 'WABCOINDIA', 'WELCORP', 'WESTLIFE',
    'WOCKPHARMA', 'ZEEL', 'ZENSARTECH', 'ZODIACLOTH'
]

def get_stocks_by_index(index_name):
    """Get list of stocks by index name"""
    index_mapping = {
        'Nifty 50': NIFTY_50,
        'Nifty 100': NIFTY_100,
        'Nifty 200': NIFTY_200,
        'Large Cap': LARGE_CAP,
        'Mid Cap': MID_CAP,
        'Small Cap': SMALL_CAP
    }
    stocks = index_mapping.get(index_name, [])
    # Remove duplicates and add .NS suffix
    unique_stocks = list(set(stocks))
    return [stock + ".NS" for stock in unique_stocks]

def get_stocks_by_sector(sector_name):
    """Get list of stocks by sector name"""
    stocks = SECTOR_STOCKS.get(sector_name, [])
    # Remove duplicates and add .NS suffix
    unique_stocks = list(set(stocks))
    return [stock + ".NS" for stock in unique_stocks]

def get_all_sectors():
    """Get list of all available sectors"""
    return list(SECTOR_STOCKS.keys())

def get_all_indices():
    """Get list of all available indices"""
    return ['Nifty 50', 'Nifty 100', 'Nifty 200', 'Large Cap', 'Mid Cap', 'Small Cap']

# Popular Indian stocks with company names for better search
STOCK_NAMES = {
    'RELIANCE': 'Reliance Industries Ltd',
    'TCS': 'Tata Consultancy Services',
    'HDFCBANK': 'HDFC Bank Ltd',
    'INFY': 'Infosys Ltd',
    'ICICIBANK': 'ICICI Bank Ltd',
    'SBIN': 'State Bank of India',
    'BHARTIARTL': 'Bharti Airtel Ltd',
    'ITC': 'ITC Ltd',
    'LT': 'Larsen & Toubro Ltd',
    'HCLTECH': 'HCL Technologies Ltd',
    'MARUTI': 'Maruti Suzuki India Ltd',
    'SUNPHARMA': 'Sun Pharmaceutical Industries',
    'BAJFINANCE': 'Bajaj Finance Ltd',
    'KOTAKBANK': 'Kotak Mahindra Bank Ltd',
    'ASIANPAINT': 'Asian Paints Ltd',
    'TITAN': 'Titan Company Ltd',
    'NESTLEIND': 'Nestle India Ltd',
    'ULTRACEMCO': 'UltraTech Cement Ltd',
    'WIPRO': 'Wipro Ltd',
    'AXISBANK': 'Axis Bank Ltd',
    'M&M': 'Mahindra & Mahindra Ltd',
    'POWERGRID': 'Power Grid Corporation',
    'NTPC': 'NTPC Ltd',
    'JSWSTEEL': 'JSW Steel Ltd',
    'TATAMOTORS': 'Tata Motors Ltd',
    'TECHM': 'Tech Mahindra Ltd',
    'HINDALCO': 'Hindalco Industries Ltd',
    'INDUSINDBK': 'IndusInd Bank Ltd',
    'BRITANNIA': 'Britannia Industries Ltd',
    'CIPLA': 'Cipla Ltd',
    'DRREDDY': 'Dr. Reddys Laboratories',
    'APOLLOHOSP': 'Apollo Hospitals Enterprise',
    'DIVISLAB': 'Divis Laboratories Ltd',
    'BAJAJ-AUTO': 'Bajaj Auto Ltd',
    'HEROMOTOCO': 'Hero MotoCorp Ltd',
    'EICHERMOT': 'Eicher Motors Ltd',
    'TATASTEEL': 'Tata Steel Ltd',
    'ADANIPORTS': 'Adani Ports and SEZ Ltd',
    'COALINDIA': 'Coal India Ltd',
    'ONGC': 'Oil & Natural Gas Corporation',
    'BPCL': 'Bharat Petroleum Corporation',
    'GRASIM': 'Grasim Industries Ltd',
    'HINDUNILVR': 'Hindustan Unilever Ltd',
    'DMART': 'Avenue Supermarts Ltd',
    'BAJAJFINSV': 'Bajaj Finserv Ltd',
    'TATACONSUM': 'Tata Consumer Products',
    'SBILIFE': 'SBI Life Insurance Company',
    'HDFCLIFE': 'HDFC Life Insurance Company',
    'LICI': 'Life Insurance Corporation',
    'POLICYBZR': 'PB Fintech Ltd',
    'ADANIENT': 'Adani Enterprises Ltd',
    'ADANIGREEN': 'Adani Green Energy Ltd',
    'GODREJCP': 'Godrej Consumer Products',
    'MARICO': 'Marico Ltd',
    'DABUR': 'Dabur India Ltd',
    'COLPAL': 'Colgate Palmolive India',
    'PIDILITIND': 'Pidilite Industries Ltd',
    'BERGEPAINT': 'Berger Paints India Ltd',
    'PAGEIND': 'Page Industries Ltd',
    'VBL': 'Varun Beverages Ltd',
    'UBL': 'United Breweries Ltd',
    'AMBUJACEM': 'Ambuja Cements Ltd',
    'SHREECEM': 'Shree Cement Ltd',
    'ACC': 'ACC Ltd',
    'JKCEMENT': 'JK Cement Ltd',
    'RAMCOCEM': 'The Ramco Cements Ltd',
    'SAIL': 'Steel Authority of India',
    'NMDC': 'NMDC Ltd',
    'VEDL': 'Vedanta Ltd',
    'JINDALSTEL': 'Jindal Steel & Power Ltd',
    'TATAPOWER': 'Tata Power Company Ltd',
    'IOC': 'Indian Oil Corporation Ltd',
    'GAIL': 'GAIL India Ltd',
    'PETRONET': 'Petronet LNG Ltd',
    'INDIGO': 'InterGlobe Aviation Ltd',
    'SPICEJET': 'SpiceJet Ltd',
    'IRCTC': 'Indian Railway Catering',
    'CONCOR': 'Container Corporation of India',
    'DLF': 'DLF Ltd',
    'GODREJPROP': 'Godrej Properties Ltd',
    'OBEROIRLTY': 'Oberoi Realty Ltd',
    'BANKBARODA': 'Bank of Baroda',
    'PNB': 'Punjab National Bank',
    'CANBK': 'Canara Bank',
    'UNIONBANK': 'Union Bank of India',
    'IDFCFIRSTB': 'IDFC First Bank Ltd',
    'FEDERALBNK': 'Federal Bank Ltd',
    'RBLBANK': 'RBL Bank Ltd',
    'YESBANK': 'Yes Bank Ltd',
    'BANDHANBNK': 'Bandhan Bank Ltd',
    'PFC': 'Power Finance Corporation',
    'RECLTD': 'REC Ltd',
    'IRFC': 'Indian Railway Finance Corporation',
    'LICHSGFIN': 'LIC Housing Finance Ltd',
    'BAJAJHLDNG': 'Bajaj Holdings & Investment',
    'ICICIPRULI': 'ICICI Prudential Life Insurance',
    'SBICARD': 'SBI Cards and Payment Services',
    'HFCL': 'HFCL Ltd',
    'TRENT': 'Trent Ltd',
    'AUROPHARMA': 'Aurobindo Pharma Ltd',
    'LUPIN': 'Lupin Ltd',
    'BIOCON': 'Biocon Ltd',
    'GLENMARK': 'Glenmark Pharmaceuticals Ltd',
    'TORNTPHARM': 'Torrent Pharmaceuticals Ltd',
    'ALKEM': 'Alkem Laboratories Ltd',
    'MPHASIS': 'Mphasis Ltd',
    'OFSS': 'Oracle Financial Services Software',
    'PERSISTENT': 'Persistent Systems Ltd',
    'LTTS': 'L&T Technology Services Ltd',
    'COFORGE': 'Coforge Ltd',
    'MINDTREE': 'Mindtree Ltd',
    'NAUKRI': 'Info Edge India Ltd',
    'JUBLFOOD': 'Jubilant FoodWorks Ltd',
    'ESCORTS': 'Escorts Ltd',
    'TVSMOTOR': 'TVS Motor Company Ltd',
    'ASHOKLEY': 'Ashok Leyland Ltd',
    'BHARATFORG': 'Bharat Forge Ltd',
    'BALKRISIND': 'Balkrishna Industries Ltd',
    'MRF': 'MRF Ltd',
    'APOLLOTYRE': 'Apollo Tyres Ltd',
    'CEAT': 'CEAT Ltd',
    'BOSCHLTD': 'Bosch Ltd',
    'MOTHERSON': 'Motherson Sumi Systems Ltd',
    'EXIDEIND': 'Exide Industries Ltd',
    'AMARAJABAT': 'Amara Raja Batteries Ltd',
    'SIEMENS': 'Siemens Ltd',
    'ABB': 'ABB India Ltd',
    'HAVELLS': 'Havells India Ltd',
    'VOLTAS': 'Voltas Ltd',
    'BLUEDART': 'Blue Dart Express Ltd',
    'DELHIVERY': 'Delhivery Ltd',
    'CROMPTON': 'Crompton Greaves Consumer Electricals',
    'VGUARD': 'V-Guard Industries Ltd',
    'WHIRLPOOL': 'Whirlpool of India Ltd',
    'DIXON': 'Dixon Technologies India Ltd',
    'AMBER': 'Amber Enterprises India Ltd'
}

def get_all_stocks():
    """Get comprehensive list of all available stocks"""
    all_stocks = set()
    
    # Add stocks from all indices
    all_stocks.update(NIFTY_200)
    all_stocks.update(MID_CAP)
    all_stocks.update(SMALL_CAP)
    
    # Add stocks from all sectors
    for sector_stocks in SECTOR_STOCKS.values():
        all_stocks.update(sector_stocks)
    
    # Add additional popular stocks
    all_stocks.update(STOCK_NAMES.keys())
    
    return sorted(list(all_stocks))

def search_stock(query):
    """Search for stocks containing the query string"""
    if not query:
        return []
        
    query = query.upper().strip()
    all_stocks = get_all_stocks()
    matching_stocks = []
    
    # Search by symbol
    for stock in all_stocks:
        if query in stock:
            matching_stocks.append(stock)
    
    # Search by company name
    for symbol, name in STOCK_NAMES.items():
        if query in name.upper() and symbol not in matching_stocks:
            matching_stocks.append(symbol)
    
    return sorted(matching_stocks)[:20]  # Limit to top 20 results

def get_stock_display_name(symbol):
    """Get display name for a stock symbol"""
    clean_symbol = symbol.replace('.NS', '')
    return STOCK_NAMES.get(clean_symbol, clean_symbol)

def get_stock_sector(symbol):
    """Get sector for a given stock symbol"""
    for sector, stocks in SECTOR_STOCKS.items():
        if symbol in stocks:
            return sector
    return 'Other'

def get_stock_market_cap_category(symbol):
    """Get market cap category for a given stock symbol"""
    if symbol in LARGE_CAP:
        return 'Large Cap'
    elif symbol in MID_CAP:
        return 'Mid Cap'
    elif symbol in SMALL_CAP:
        return 'Small Cap'
    else:
        return 'Unknown'
