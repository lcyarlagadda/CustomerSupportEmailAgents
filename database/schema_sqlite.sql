-- TaskFlow Pro Customer Database Schema
-- SQLite 3.x (Built-in to Python)

-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active', -- active, suspended, cancelled
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_status ON customers(status);

-- Subscriptions table
CREATE TABLE IF NOT EXISTS subscriptions (
    subscription_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    plan_name TEXT NOT NULL, -- Starter, Pro, Enterprise
    plan_price REAL NOT NULL,
    billing_cycle TEXT NOT NULL, -- monthly, yearly
    status TEXT NOT NULL DEFAULT 'active', -- active, cancelled, past_due
    is_active INTEGER NOT NULL DEFAULT 1,
    next_billing_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_customer ON subscriptions(customer_id);
CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
CREATE INDEX IF NOT EXISTS idx_subscriptions_active ON subscriptions(customer_id, is_active);

-- Payments table
CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    payment_method TEXT NOT NULL, -- card, paypal, bank_transfer
    last_payment_date TEXT,
    last_payment_amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_payments_customer ON payments(customer_id);

-- Invoices table
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    invoice_number TEXT UNIQUE NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL, -- draft, open, paid, void
    created_at TEXT NOT NULL,
    due_date TEXT NOT NULL,
    paid_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices(customer_id);
CREATE INDEX IF NOT EXISTS idx_invoices_status ON invoices(status);
CREATE INDEX IF NOT EXISTS idx_invoices_number ON invoices(invoice_number);

-- Usage tracking table
CREATE TABLE IF NOT EXISTS usage (
    usage_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    current_usage TEXT NOT NULL, -- JSON: {"projects": 5, "team_members": 3, "storage_gb": 25}
    plan_limits TEXT NOT NULL, -- JSON: {"projects": 10, "team_members": 5, "storage_gb": 50}
    reset_date TEXT NOT NULL, -- When usage resets (usually billing date)
    updated_at TEXT NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE INDEX IF NOT EXISTS idx_usage_customer ON usage(customer_id);

-- Sample data for testing
INSERT OR IGNORE INTO customers (customer_id, email, name, status, created_at) VALUES
('cust_sample_001', 'john.doe@example.com', 'John Doe', 'active', datetime('now', '-180 days')),
('cust_sample_002', 'jane.smith@example.com', 'Jane Smith', 'active', datetime('now', '-90 days')),
('cust_sample_003', 'test@example.com', 'Test User', 'active', datetime('now', '-30 days'));

INSERT OR IGNORE INTO subscriptions (subscription_id, customer_id, plan_name, plan_price, billing_cycle, next_billing_date, created_at)
SELECT 
    'sub_001',
    'cust_sample_001',
    'Pro Plan',
    49.99,
    'monthly',
    datetime('now', '+15 days'),
    datetime('now', '-180 days')
WHERE NOT EXISTS (SELECT 1 FROM subscriptions WHERE subscription_id = 'sub_001');

INSERT OR IGNORE INTO payments (payment_id, customer_id, payment_method, last_payment_date, last_payment_amount)
SELECT 
    'pay_001',
    'cust_sample_001',
    'Visa ending in 4242',
    datetime('now', '-15 days'),
    49.99
WHERE NOT EXISTS (SELECT 1 FROM payments WHERE payment_id = 'pay_001');

INSERT OR IGNORE INTO usage (usage_id, customer_id, current_usage, plan_limits, reset_date, updated_at)
SELECT 
    'usage_001',
    'cust_sample_001',
    '{"projects": 8, "team_members": 5, "storage_gb": 45}',
    '{"projects": 10, "team_members": 10, "storage_gb": 100}',
    datetime('now', '+15 days'),
    datetime('now')
WHERE NOT EXISTS (SELECT 1 FROM usage WHERE usage_id = 'usage_001');

-- Sample invoices
INSERT OR IGNORE INTO invoices (invoice_id, customer_id, invoice_number, amount, status, created_at, due_date, paid_date)
SELECT 
    'inv_001',
    'cust_sample_001',
    'INV-2024-001',
    49.99,
    'paid',
    datetime('now', '-30 days'),
    datetime('now', '-23 days'),
    datetime('now', '-28 days')
WHERE NOT EXISTS (SELECT 1 FROM invoices WHERE invoice_id = 'inv_001');

INSERT OR IGNORE INTO invoices (invoice_id, customer_id, invoice_number, amount, status, created_at, due_date, paid_date)
SELECT 
    'inv_002',
    'cust_sample_001',
    'INV-2024-002',
    49.99,
    'paid',
    datetime('now', '-60 days'),
    datetime('now', '-53 days'),
    datetime('now', '-58 days')
WHERE NOT EXISTS (SELECT 1 FROM invoices WHERE invoice_id = 'inv_002');

INSERT OR IGNORE INTO invoices (invoice_id, customer_id, invoice_number, amount, status, created_at, due_date, paid_date)
SELECT 
    'inv_003',
    'cust_sample_001',
    'INV-2024-003',
    49.99,
    'paid',
    datetime('now', '-90 days'),
    datetime('now', '-83 days'),
    datetime('now', '-88 days')
WHERE NOT EXISTS (SELECT 1 FROM invoices WHERE invoice_id = 'inv_003');

