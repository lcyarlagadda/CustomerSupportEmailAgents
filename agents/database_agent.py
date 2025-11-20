"""
Database Agent for Customer Data Retrieval (SQLite)

This agent queries the database for customer-specific information:
- Account status and details
- Subscription/membership information
- Billing history and invoices
- Payment methods
- Usage statistics

Uses SQLite (built-in, perfect for Colab) with optional Redis caching.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import sqlite3
import os


@dataclass
class CustomerAccount:
    """Customer account information."""
    customer_id: str
    email: str
    name: str
    status: str  # active, suspended, cancelled
    created_at: datetime
    
    # Subscription info
    plan_name: str
    plan_price: float
    billing_cycle: str  # monthly, yearly
    next_billing_date: datetime
    
    # Payment info
    payment_method: str  # card, paypal, etc.
    last_payment_date: Optional[datetime]
    last_payment_amount: Optional[float]
    
    # Usage
    current_usage: Dict[str, Any]
    plan_limits: Dict[str, Any]


class DatabaseAgent:
    """
    SQLite-based database agent.
    
    Retrieves real-time customer data for billing, account, and membership queries.
    Perfect for development, testing, and Colab environments.
    
    SECURITY: This agent only executes SELECT queries (read-only).
    No INSERT, UPDATE, DELETE, or other write operations are permitted.
    All queries are parameterized to prevent SQL injection.
    """
    
    def __init__(self, use_cache: bool = True, db_path: str = "customer_data.db"):
        """
        Initialize database agent.
        
        Args:
            use_cache: Enable Redis caching for frequent queries
            db_path: Path to SQLite database file
        """
        self.use_cache = use_cache
        self.db_path = db_path
        self.db_connection = None
        self.cache = None
        
        # Initialize connections
        self._initialize_database()
        if use_cache:
            self._initialize_cache()
    
    def _initialize_database(self):
        """Initialize SQLite database and create tables."""
        try:
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self.db_connection.row_factory = sqlite3.Row
            
            # Create tables
            cursor = self.db_connection.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    customer_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    name TEXT,
                    status TEXT DEFAULT 'active',
                    created_at TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    subscription_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    plan_name TEXT,
                    plan_price REAL,
                    billing_cycle TEXT,
                    next_billing_date TEXT,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS payments (
                    payment_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    payment_method TEXT,
                    last_payment_date TEXT,
                    last_payment_amount REAL,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage (
                    usage_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    current_usage TEXT,
                    plan_limits TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS invoices (
                    invoice_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    invoice_number TEXT,
                    amount REAL,
                    status TEXT,
                    created_at TEXT,
                    due_date TEXT,
                    paid_date TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(customer_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id)")
            
            self.db_connection.commit()
            print("Database Agent: SQLite connected")
            
            # Add sample data if database is empty
            self._add_sample_data_if_empty()
            
        except Exception as e:
            print(f"Warning: Database Agent: SQLite initialization failed ({e})")
            self.db_connection = None
    
    def _add_sample_data_if_empty(self):
        """Add sample customer data if database is empty."""
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("  Adding sample customer data...")
            
            # Sample customer
            customer_id = "cust_sample_001"
            email = "john.doe@example.com"
            
            cursor.execute("""
                INSERT INTO customers (customer_id, email, name, status, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (customer_id, email, "John Doe", "active", 
                  (datetime.now() - timedelta(days=180)).isoformat()))
            
            cursor.execute("""
                INSERT INTO subscriptions 
                (subscription_id, customer_id, plan_name, plan_price, billing_cycle, 
                 next_billing_date, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ("sub_001", customer_id, "Pro Plan", 49.99, "monthly",
                  (datetime.now() + timedelta(days=15)).isoformat(), 1))
            
            cursor.execute("""
                INSERT INTO payments 
                (payment_id, customer_id, payment_method, last_payment_date, last_payment_amount)
                VALUES (?, ?, ?, ?, ?)
            """, ("pay_001", customer_id, "Visa ending in 4242",
                  (datetime.now() - timedelta(days=15)).isoformat(), 49.99))
            
            cursor.execute("""
                INSERT INTO usage (usage_id, customer_id, current_usage, plan_limits)
                VALUES (?, ?, ?, ?)
            """, ("usage_001", customer_id,
                  json.dumps({"projects": 8, "team_members": 5, "storage_gb": 45}),
                  json.dumps({"projects": 10, "team_members": 10, "storage_gb": 100})))
            
            # Add sample invoices
            for i in range(1, 4):
                cursor.execute("""
                    INSERT INTO invoices 
                    (invoice_id, customer_id, invoice_number, amount, status, 
                     created_at, due_date, paid_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (f"inv_00{i}", customer_id, f"INV-2024-00{i}", 49.99, "paid",
                      (datetime.now() - timedelta(days=30 * i)).isoformat(),
                      (datetime.now() - timedelta(days=30 * i - 7)).isoformat(),
                      (datetime.now() - timedelta(days=30 * i - 2)).isoformat()))
            
            self.db_connection.commit()
            print(f"  Sample customer added: {email}")
    
    def _initialize_cache(self):
        """Initialize Redis cache."""
        try:
            import redis
            
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            
            self.cache = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=0,
                decode_responses=True
            )
            self.cache.ping()
            print(f"[Database] Redis cache connected ({redis_host}:{redis_port})")
            
        except ImportError:
            print("[Database] Redis not installed - caching disabled (install: pip install redis)")
            self.cache = None
        except Exception as e:
            print(f"[Database] Redis connection failed - caching disabled")
            print(f"[Database]   Error: {e}")
            print(f"[Database]   Tip: Check REDIS_HOST and REDIS_PORT in .env or start Redis server")
            self.cache = None
    
    def should_check_database(self, category: str, query: str) -> bool:
        """
        Determine if query requires database lookup.
        
        Args:
            category: Email category
            query: Customer query
            
        Returns:
            True if database lookup needed
        """
        # Always check DB for billing category
        if category == "billing":
            return True
        
        # Check for account-related keywords
        account_keywords = [
            "subscription", "plan", "billing", "invoice", "payment",
            "account", "membership", "upgrade", "downgrade", "cancel",
            "charge", "refund", "receipt", "transaction", "balance",
            "usage", "limit", "quota", "tier"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in account_keywords)
    
    def get_customer_by_email(self, email: str) -> Optional[CustomerAccount]:
        """
        Retrieve customer account by email.
        
        Args:
            email: Customer email address
            
        Returns:
            CustomerAccount or None if not found
        """
        print(f"\n[Database] Checking database for customer: {email}")
        
        # Check cache first
        if self.cache:
            try:
                cached = self.cache.get(f"customer:{email}")
                if cached:
                    print(f"[Database] Cache hit - returning cached data for {email}")
                    data = json.loads(cached)
                    return self._dict_to_account(data)
                else:
                    print(f"[Database] Cache miss - querying database")
            except Exception as e:
                print(f"[Database] Cache error: {e}")
                pass
        
        # Query database
        if self.db_connection:
            try:
                print(f"[Database] Executing SELECT query for {email}")
                cursor = self.db_connection.cursor()
                
                cursor.execute("""
                    SELECT 
                        c.customer_id,
                        c.email,
                        c.name,
                        c.status,
                        c.created_at,
                        s.plan_name,
                        s.plan_price,
                        s.billing_cycle,
                        s.next_billing_date,
                        p.payment_method,
                        p.last_payment_date,
                        p.last_payment_amount,
                        u.current_usage,
                        u.plan_limits
                    FROM customers c
                    LEFT JOIN subscriptions s ON c.customer_id = s.customer_id
                    LEFT JOIN payments p ON c.customer_id = p.customer_id
                    LEFT JOIN usage u ON c.customer_id = u.customer_id
                    WHERE c.email = ? AND s.is_active = 1
                    LIMIT 1
                """, (email,))
                
                row = cursor.fetchone()
                
                if row:
                    account = CustomerAccount(
                        customer_id=row['customer_id'],
                        email=row['email'],
                        name=row['name'],
                        status=row['status'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        plan_name=row['plan_name'],
                        plan_price=float(row['plan_price']),
                        billing_cycle=row['billing_cycle'],
                        next_billing_date=datetime.fromisoformat(row['next_billing_date']),
                        payment_method=row['payment_method'],
                        last_payment_date=datetime.fromisoformat(row['last_payment_date']) if row['last_payment_date'] else None,
                        last_payment_amount=float(row['last_payment_amount']) if row['last_payment_amount'] else None,
                        current_usage=json.loads(row['current_usage']),
                        plan_limits=json.loads(row['plan_limits'])
                    )
                    
                    print(f"[Database] Found customer: {account.name}")
                    print(f"[Database]   Plan: {account.plan_name} (${account.plan_price}/{account.billing_cycle})")
                    print(f"[Database]   Status: {account.status}")
                    print(f"[Database]   Next billing: {account.next_billing_date.strftime('%Y-%m-%d')}")
                    
                    # Cache result
                    if self.cache:
                        try:
                            cache_data = self._account_to_dict(account)
                            self.cache.setex(
                                f"customer:{email}",
                                300,  # 5 minutes TTL
                                json.dumps(cache_data, default=str)
                            )
                        except Exception:
                            pass
                    
                    return account
                else:
                    print(f"[Database] No customer found for {email}")
                    return None
                
            except Exception as e:
                print(f"[Database] Query error: {e}")
                return None
        else:
            print(f"[Database] Database not connected, skipping lookup")
            return None
    
    def get_recent_invoices(self, customer_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent invoices for a customer (READ-ONLY).
        
        Args:
            customer_id: Customer ID
            limit: Number of invoices to retrieve
            
        Returns:
            List of invoice dictionaries
        """
        if not self.db_connection:
            return []
        
        try:
            print(f"[Database] Fetching {limit} recent invoices for customer {customer_id}")
            cursor = self.db_connection.cursor()
            
            # READ-ONLY: SELECT query only
            cursor.execute("""
                SELECT 
                    invoice_id,
                    invoice_number,
                    amount,
                    status,
                    created_at,
                    due_date,
                    paid_date
                FROM invoices
                WHERE customer_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (customer_id, limit))
            
            invoices = [dict(row) for row in cursor.fetchall()]
            print(f"[Database] Found {len(invoices)} invoices")
            return invoices
            
        except Exception as e:
            print(f"[Database] Invoice query error: {e}")
            return []
    
    def format_account_context(self, account: CustomerAccount) -> str:
        """
        Format account information for LLM context.
        
        Args:
            account: Customer account
            
        Returns:
            Formatted context string
        """
        # Calculate days until next billing
        days_until_billing = (account.next_billing_date - datetime.now()).days
        
        # Format usage info
        usage_info = []
        for key, value in account.current_usage.items():
            limit = account.plan_limits.get(key, "unlimited")
            usage_info.append(f"{key}: {value}/{limit}")
        
        context = f"""
CUSTOMER ACCOUNT INFORMATION:
- Account Status: {account.status.upper()}
- Customer Name: {account.name}
- Email: {account.email}
- Account Created: {account.created_at.strftime('%Y-%m-%d')}

SUBSCRIPTION:
- Current Plan: {account.plan_name}
- Price: ${account.plan_price}/{account.billing_cycle}
- Next Billing Date: {account.next_billing_date.strftime('%Y-%m-%d')} ({days_until_billing} days)
- Billing Cycle: {account.billing_cycle}

PAYMENT:
- Payment Method: {account.payment_method}
- Last Payment: ${account.last_payment_amount or 0:.2f} on {account.last_payment_date.strftime('%Y-%m-%d') if account.last_payment_date else 'N/A'}

USAGE:
{chr(10).join(f"- {info}" for info in usage_info)}

Use this information to answer the customer's billing or account question accurately.
If they ask about charges, refer to their current plan price.
If they ask about billing date, provide the exact date shown above.
"""
        return context.strip()
    
    def _dict_to_account(self, data: Dict) -> CustomerAccount:
        """Convert dictionary to CustomerAccount."""
        return CustomerAccount(
            customer_id=data['customer_id'],
            email=data['email'],
            name=data['name'],
            status=data['status'],
            created_at=datetime.fromisoformat(data['created_at']),
            plan_name=data['plan_name'],
            plan_price=data['plan_price'],
            billing_cycle=data['billing_cycle'],
            next_billing_date=datetime.fromisoformat(data['next_billing_date']),
            payment_method=data['payment_method'],
            last_payment_date=datetime.fromisoformat(data['last_payment_date']) if data.get('last_payment_date') else None,
            last_payment_amount=data.get('last_payment_amount'),
            current_usage=data['current_usage'],
            plan_limits=data['plan_limits']
        )
    
    def _account_to_dict(self, account: CustomerAccount) -> Dict:
        """Convert CustomerAccount to dictionary for caching."""
        return {
            'customer_id': account.customer_id,
            'email': account.email,
            'name': account.name,
            'status': account.status,
            'created_at': account.created_at.isoformat(),
            'plan_name': account.plan_name,
            'plan_price': account.plan_price,
            'billing_cycle': account.billing_cycle,
            'next_billing_date': account.next_billing_date.isoformat(),
            'payment_method': account.payment_method,
            'last_payment_date': account.last_payment_date.isoformat() if account.last_payment_date else None,
            'last_payment_amount': account.last_payment_amount,
            'current_usage': account.current_usage,
            'plan_limits': account.plan_limits
        }
    
    def close(self):
        """Close database connections."""
        if self.db_connection:
            self.db_connection.close()
        if self.cache:
            self.cache.close()


# Global instance
_database_agent = None


def get_database_agent() -> DatabaseAgent:
    """Get or create the global database agent instance."""
    global _database_agent
    if _database_agent is None:
        _database_agent = DatabaseAgent(use_cache=True)
    return _database_agent

