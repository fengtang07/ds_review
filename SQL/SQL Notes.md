# SQL Notes for Data Science

# **Chapter I: The Basics of SQL Querying**

### **1.1 Database and Table Management (DDL)**

```sql
# Create a database if it doesn't already exist
CREATE DATABASE IF NOT EXISTS sql_store2;

# Switch to the newly created database
USE sql_store2;

# Create a table with various constraints
CREATE TABLE IF NOT EXISTS customers
(
	customer_id INT PRIMARY KEY AUTO_INCREMENT,
	first_name VARCHAR(50) NOT NULL,
	points INT NOT NULL DEFAULT 0,
	email VARCHAR(255) NOT NULL UNIQUE
);

# Modify an existing table
ALTER TABLE customers
	ADD last_name VARCHAR(50) NOT NULL AFTER first_name,
	MODIFY COLUMN first_name VARCHAR(50) DEFAULT '',
	DROP points;

# Add/Drop keys and constraints
ALTER TABLE orders
	ADD PRIMARY KEY (order_id),
	DROP PRIMARY KEY,
	DROP FOREIGN KEY fk_orders_customers,
	ADD FOREIGN KEY fk_orders_customers (customer_id)
		REFERENCES customers (customer_id)
		ON UPDATE CASCADE -- If a customer_id is updated in `customers`, update it here too.
		ON DELETE NO ACTION; -- Prevent deleting a customer if they have orders.

# Define relationships during table creation
DROP TABLE IF EXISTS orders;
CREATE TABLE orders
(
	order_id INT PRIMARY KEY,
	customer_id INT NOT NULL,
	FOREIGN KEY fk_orders_customers(customer_id)
		REFERENCES customers (customer_id)
		ON UPDATE CASCADE
		ON DELETE NO ACTION
);
```

### **1.2 Joining Tables**

Joins are used to combine rows from two or more tables based on a related column between them.

| **Join Type** | **Description** | **Use Case** |
| --- | --- | --- |
| **INNER JOIN** | Returns records that have matching values in both tables. | Get orders that belong to existing customers. |
| **LEFT JOIN** | Returns all records from the left table, and the matched records from the right table. | Find all customers, even those who haven't placed an order. |
| **RIGHT JOIN** | Returns all records from the right table, and the matched records from the left table. | Find all products, even if they've never been ordered. |
| **FULL OUTER JOIN** | Returns all records when there is a match in either left or right table. *MySQL doesn't have this, but it can be emulated with `UNION`*. | Get a complete list of all customers and all products, matched where possible. |
| **SELF JOIN** | A regular join, but the table is joined with itself. | Find employees who have the same manager. |
| **CROSS JOIN** | Returns the Cartesian product of the two tables (all possible combinations). | Generate a list of all possible pairings of customers and products. |

```sql
-- Example: Find all customers and the number of orders they've placed.
-- Customers with zero orders will be included.
SELECT
    c.customer_id,
    c.first_name,
    COUNT(o.order_id) AS number_of_orders
FROM
    customers c
LEFT JOIN
    orders o ON c.customer_id = o.customer_id
GROUP BY
    c.customer_id, c.first_name;

-- Example: SELF JOIN to find an employee and their manager
SELECT
    e.name AS employee_name,
    m.name AS manager_name
FROM
    employees e
LEFT JOIN
    employees m ON e.manager_id = m.employee_id;
```

### **1.3 Subqueries**

A **correlated subquery** is a subquery that uses values from the **outer query**. It is evaluated **once for each row** of the outer query.

**Row-by-Row Evaluation**

```sql
-- Select invoices that are larger than the average for that specific client
SELECT *
FROM invoices i
WHERE invoice_total > (
    SELECT AVG(invoice_total)
    FROM invoices
    WHERE client_id = i.client_id -- The subquery is "correlated" with the outer query via i.client_id
);
```

- `i.client_id` refers to the outer query.
- For each invoice row, the subquery calculates the **average invoice_total** for that specific `client_id`.

**Use Cases**

- Comparing a value against a group-specific aggregate.
- Validating row-level conditions against contextual data.
- Row-dependent filtering logic.

**Common Confusions & Why** | Confusion | Why It Happens | | :--- | :--- | | `client_id = i.client_id` looks self-referencing | Same column name used inside and outside can be misleading. | | Feels like infinite loop | Correlated subqueries run once **per outer row**, not recursively. | | Hard to debug | Query logic is implicit; harder to trace row-by-row behavior. |

**Alternatives** You can often rewrite correlated subqueries using `JOIN` + `GROUP BY`, which is often more performant.

```sql
SELECT i.*
FROM invoices i
JOIN (
    SELECT client_id, AVG(invoice_total) AS avg_total
    FROM invoices
    GROUP BY client_id
) avg_invoices ON i.client_id = avg_invoices.client_id
WHERE i.invoice_total > avg_invoices.avg_total;
```

### **1.4 Common Table Expressions (CTEs)**

A **Common Table Expression (CTE)** is a temporary, named result set that you can reference within a `SELECT`, `INSERT`, `UPDATE`, or `DELETE` statement. CTEs help break down complex queries into simple, logical, and readable steps.

| **Benefit** | **Description** |
| --- | --- |
| **Readability** | Organizes long queries into logical blocks, like variables in programming. |
| **Modularity** | You can define a CTE once and reference it multiple times in the main query. |
| **Recursion** | CTEs can reference themselves to solve recursive problems (e.g., organizational hierarchies). |

```sql
-- Using a CTE to find customers who spent more than the overall average.
WITH CustomerSpending AS (
    -- First, calculate total spending for each customer
    SELECT
        customer_id,
        SUM(payment_total) AS total_spent
    FROM invoices
    GROUP BY customer_id
),
AverageSpending AS (
    -- Second, calculate the overall average spending
    SELECT AVG(total_spent) AS avg_spent FROM CustomerSpending
)
-- Finally, select customers who spent more than the average
SELECT
    c.first_name,
    cs.total_spent
FROM
    customers c
JOIN
    CustomerSpending cs ON c.customer_id = cs.customer_id
WHERE
    cs.total_spent > (SELECT avg_spent FROM AverageSpending);
```

### **1.5 Aggregation and Grouping**

Aggregation functions perform a calculation on a set of rows and return a single summary value. `GROUP BY` is used to arrange identical data into groups.

| **Function** | **Description** |
| --- | --- |
| `COUNT(column)` | Counts the number of non-null rows. `COUNT(*)` counts all rows. |
| `SUM(column)` | Calculates the sum of values. |
| `AVG(column)` | Calculates the average of values. |
| `MIN(column)` | Finds the minimum value. |
| `MAX(column)` | Finds the maximum value. |
- **`GROUP BY` clause**: Used with aggregate functions to group rows that have the same values in specified columns into summary rows.
- **`HAVING` clause**: Used to filter groups based on the results of aggregate functions. It's like a `WHERE` clause for `GROUP BY`.

```sql
-- Select the number of customers and average points for each state,
-- but only include states with more than 10 customers.
SELECT
    state,
    COUNT(customer_id) AS number_of_customers,
    AVG(points) AS average_points
FROM
    customers
WHERE
    state IS NOT NULL
GROUP BY
    state
HAVING
    COUNT(customer_id) > 10
ORDER BY
    number_of_customers DESC;
```

# **Chapter II: Advanced Querying Techniques**

### **2.1 Window Functions**

Window functions perform calculations across a set of table rows that are somehow related to the current row. Unlike aggregate functions, they do not collapse rows; they return a value for each row.

The syntax involves the `OVER()` clause, which defines the "window" of rows to operate on.

- `PARTITION BY`: Divides rows into partitions (like `GROUP BY` but for windows).
- `ORDER BY`: Orders rows within each partition.
- `ROWS BETWEEN ...`: Specifies the frame of rows to include in the window (e.g., `ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING`).

| **Function Type** | **Function** | **Description** |
| --- | --- | --- |
| **Ranking** | `ROW_NUMBER()` | Assigns a unique number to each row in the partition. |
|  | `RANK()` | Assigns a rank. Rows with the same value get the same rank, with gaps. (e.g., 1, 2, 2, 4) |
|  | `DENSE_RANK()` | Assigns a rank without gaps. (e.g., 1, 2, 2, 3) |
| **Offset** | `LEAD(col, n)` | Accesses data from a row `n` rows *after* the current row. |
|  | `LAG(col, n)` | Accesses data from a row `n` rows *before* the current row. |
| **Aggregate** | `SUM() OVER(...)` | Calculates a running total or cumulative sum. |
|  | `AVG() OVER(...)` | Calculates a moving average. |

```sql
-- Example: Rank products by price within each category and find the next most expensive product.
WITH ProductRanks AS (
    SELECT
        product_name,
        category,
        price,
        RANK() OVER (PARTITION BY category ORDER BY price DESC) AS price_rank,
        LEAD(product_name, 1) OVER (PARTITION BY category ORDER BY price DESC) AS next_most_expensive
    FROM
        products
)
SELECT * FROM ProductRanks WHERE price_rank <= 3;

-- Example: Calculate the 7-day moving average of sales
SELECT
    sale_date,
    daily_revenue,
    AVG(daily_revenue) OVER (ORDER BY sale_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS seven_day_moving_avg
FROM
    daily_sales;
```

### **2.2 Conditional Logic and Data Cleaning**

These functions are essential for transforming, cleaning, and preparing data for analysis.

| **Function** | **Description** | **Use Case** |
| --- | --- | --- |
| `CASE ... END` | An `if-then-else` statement for your queries. | Creating custom categories, binning data, or applying conditional logic. |
| `COALESCE(val1, val2, ...)` | Returns the first non-NULL value in a list. | Filling in missing data with a default or fallback value. |
| `NULLIF(expr1, expr2)` | Returns `NULL` if the two expressions are equal; otherwise, returns the first expression. | Preventing division-by-zero errors by converting a zero divisor to `NULL`. |

```sql
-- Example: Use CASE to categorize customers based on their points.
SELECT
    customer_id,
    points,
    CASE
        WHEN points > 3000 THEN 'Gold'
        WHEN points BETWEEN 1000 AND 3000 THEN 'Silver'
        ELSE 'Bronze'
    END AS customer_tier
FROM
    customers;

-- Example: Use COALESCE to show a default shipping date.
SELECT
    order_id,
    COALESCE(shipped_date, 'Not Shipped Yet') AS shipping_status
FROM
    orders;

-- Example: Use NULLIF to avoid division by zero.
SELECT
    total_sales / NULLIF(number_of_transactions, 0) AS avg_sale_value
FROM
    daily_summary;
```

# **Chapter III: Core Functions**

### **3.1 Numeric Functions**

| **Function** | **Description** | **Example** |
| --- | --- | --- |
| `CEILING(x)` | Rounds *up* to the nearest integer | `CEILING(4.2)` → `5` |
| `FLOOR(x)` | Rounds *down* to the nearest integer | `FLOOR(4.8)` → `4` |
| `RAND()` | Returns a random float between 0 and 1 | e.g., `0.72843` |

### **3.2 String Functions**

| **Function** | **Description** | **Example** |
| --- | --- | --- |
| `UPPER(str)` | Converts to uppercase | `UPPER('sql')` → `'SQL'` |
| `LOWER(str)` | Converts to lowercase | `LOWER('SQL')` → `'sql'` |
| `TRIM(str)` | Removes leading and trailing spaces | `TRIM(' sql ')` → `'sql'` |
| `LTRIM(str)` | Removes leading spaces | `LTRIM(' sql')` → `'sql'` |
| `RTRIM(str)` | Removes trailing spaces | `RTRIM('sql ')` → `'sql'` |
| `LEFT(str, n)` | Gets first *n* characters | `LEFT('Kingder', 4)` → `'King'` |
| `SUBSTRING(str, start, length)` | Extracts substring | `SUBSTRING('kindergarden', 3, 5)` → `'nderg'` |
| `LOCATE(substr, str)` | Finds first position of substring | `LOCATE('n', 'kindergarden')` → `3` |
| `REPLACE(str, from, to)` | Replaces substrings | `REPLACE('kindergarden', 'garden', 'garten')` → `'kindergarten'` |

### **3.3 Date Functions**

| **Function** | **Description** | **Example** |
| --- | --- | --- |
| `NOW()` | Returns current date and time | `2025-07-20 21:30:00` |
| `CURDATE()` | Returns current date only | `2025-07-20` |
| `CURTIME()` | Returns current time only | `21:30:00` |
| `YEAR(NOW())` | Extracts year | `2025` |
| `DAYNAME(NOW())` | Returns weekday name | `'Sunday'` |
| `EXTRACT(YEAR FROM NOW())` | Same as above | `2025` |

### **3.4 Date Formatting**

| **Function** | **Description** | **Example** |
| --- | --- | --- |
| `DATE_FORMAT(date, format)` | Formats date | `DATE_FORMAT(NOW(), '%M %Y')` → `'July 2025'` |
| `TIME_FORMAT(time, format)` | Formats time | `TIME_FORMAT(NOW(), '%H')` → `'21'` (24h format) |

### **3.5 Date Calculations**

| **Function** | **Description** | **Example** |
| --- | --- | --- |
| `DATE_ADD(date, INTERVAL n unit)` | Adds time to a date | `DATE_ADD(NOW(), INTERVAL 1 DAY)` |
| `DATE_SUB(date, INTERVAL n unit)` | Subtracts time from a date | `DATE_SUB(NOW(), INTERVAL 1 DAY)` |
| `DATEDIFF(date1, date2)` | Returns number of days between two dates | `DATEDIFF('2019-01-05', NOW())` |
| `TIME_TO_SEC(time)` | Converts time to total seconds | `TIME_TO_SEC('09:00') - TIME_TO_SEC(NOW())` gives time remaining in seconds |

# **Chapter IV: Database Programmability**

### **4.1 Views**

A **View** is a virtual table based on the result-set of an SQL statement. It contains rows and columns, just like a real table, but is a stored query that can be used for security, simplicity, and consistency.

```sql
-- Create a view or replace it if it already exists
CREATE OR REPLACE VIEW Brazil_Customers AS
SELECT CustomerName, ContactName
FROM Customers
WHERE Country = 'Brazil';

-- Drop a view
DROP VIEW Brazil_Customers;

-- The WITH CHECK OPTION ensures that any INSERT or UPDATE performed
-- through the view must satisfy the view’s WHERE clause.
CREATE OR REPLACE VIEW High_Value_Invoices AS
SELECT *
FROM invoices
WHERE invoice_total > 500
WITH CHECK OPTION;
```

### **4.2 Stored Procedures and Functions**

A **stored procedure** is a prepared SQL code that you can save, so the code can be reused over and over again. A **function** is similar but must return a single value.

**Stored Procedures**

```sql
-- Corrected from "CREATE FUNTION" to "CREATE PROCEDURE" as it performs actions, not returns a value.
DELIMITER $$
CREATE PROCEDURE get_invoices_with_balance()
BEGIN
    SELECT *
    FROM invoices
    WHERE invoice_total - payment_total > 0;
END$$
DELIMITER ;

-- Call the procedure
CALL get_invoices_with_balance();

-- Procedure with parameters (and using IFNULL to handle NULL inputs)
DELIMITER $$
CREATE PROCEDURE get_payments(client_id INT, payment_method_id TINYINT)
BEGIN
    SELECT *
    FROM payments p
    WHERE
        p.client_id = IFNULL(client_id, p.client_id) AND
        p.payment_method_id = IFNULL(payment_method_id, p.payment_method_id);
END$$
DELIMITER ;

-- Call with different parameters
CALL get_payments(NULL, NULL); -- Gets all payments
CALL get_payments(3, 1); -- Gets payments for client 3 via method 1
```

**Deterministic Functions** In **SQL**, when defining a **user-defined function**, the term **DETERMINISTIC** (or **NOT DETERMINISTIC**) is used to tell the database whether the function **always returns the same result for the same input**.

- **Deterministic:** `SUM()`, `UCASE()`. `UCASE('sql')` is always 'SQL'.
- **Non-Deterministic:** `NOW()`, `RAND()`. `NOW()` is different every time you call it.

### **4.3 Triggers**

A **trigger** is a stored procedure that **automatically executes** in response (before or after) to **data changes** (like `INSERT`, `UPDATE`, or `DELETE`) on a table.

| **Use Case** | **Example** |
| --- | --- |
| Audit logging | Track changes to sensitive data |
| Automatic calculations | Update totals when order lines change |
| Enforce rules | Block changes if conditions aren’t met |
| Cascading updates | Sync changes across related tables |

```sql
DELIMITER $$
CREATE TRIGGER after_delete_payment
	AFTER DELETE ON payments
    FOR EACH ROW
BEGIN
	-- OLD refers to the row that was just deleted
	UPDATE invoices
    SET payment_total = payment_total - OLD.amount
    WHERE invoice_id = OLD.invoice_id;

    INSERT INTO payments_audit
    VALUES (OLD.client_id, OLD.date, OLD.amount, 'DELETE', NOW());
END $$
DELIMITER ;
```

### **4.4 Events**

An **event** is a task that runs according to a schedule. It's like a cron job for your database.

```sql
DELIMITER $$
CREATE EVENT yearly_delete_stale_audit_rows
ON SCHEDULE
	EVERY 1 YEAR STARTS '2019-01-01' ENDS '2029-01-01'
DO BEGIN
	DELETE FROM payments_audit
    WHERE action_date < NOW() - INTERVAL 1 YEAR;
END $$
DELIMITER ;

-- Enable or disable an event
ALTER EVENT yearly_delete_stale_audit_rows ENABLE;
ALTER EVENT yearly_delete_stale_audit_rows DISABLE;
```

# **Chapter V: Core Database Concepts**

### **5.1 Transactions and Concurrency**

**ACID Properties** ACID is a set of properties that ensure reliable processing of database transactions.

| **Property** | **Mantra** | **Description** |
| --- | --- | --- |
| **Atomicity** | “All or nothing” | A transaction is **atomic**: it either **completes fully** or **not at all**. If any part fails, the entire transaction is rolled back. |
| **Consistency** | “Rules must hold” | A transaction must bring the database from one **valid state to another**, enforcing all rules, constraints, and triggers. |
| **Isolation** | “Transactions don’t interfere” | Concurrent transactions must not interfere with each other. It appears as if each transaction is **executed in isolation**. |
| **Durability** | “Once done, always done” | Once a transaction is **committed**, the change is **permanent** — even if the system crashes. |

**Transactions** A **transaction** in SQL is a sequence of operations that are **executed as a single logical unit** of work. It must **follow ACID properties**.

```sql
START TRANSACTION;

-- The following two operations will either both succeed or both fail.
INSERT INTO orders (customer_id, order_date, status)
VALUES (1, '2019-01-01', 1);

INSERT INTO order_items
VALUES (LAST_INSERT_ID(), 1, 1, 1);

COMMIT; -- Make the changes permanent

-- Or, if something went wrong:
-- ROLLBACK;
```

## Database Isolation Levels

| Isolation Level | Analogy | Phenomena Allowed |
| --- | --- | --- |
| Read Uncommitted | You read a note someone's still writing | Dirty Reads, Non-Repeatable Reads, Phantoms |
| Read Committed | You read only published notes | Non-Repeatable Reads, Phantoms |
| Repeatable Read | You lock a book once you read it | Phantoms |
| Serializable | You lock the whole library section | None |

The table shows the four standard database isolation levels, each with a helpful analogy and the concurrency phenomena

```sql
-- View and change the transaction isolation level for the current session
SHOW VARIABLES LIKE 'transaction_isolation';
SET SESSION TRANSACTION ISOLATION LEVEL SERIALIZABLE;
```

### **5.2 Data Types**

**STRING Data Types**

| Data Type | Max Length | Storage Notes | Use Case |
| --- | --- | --- | --- |
| `CHAR(x)` | 255 chars | Fixed-size, right-padded | Country codes, fixed flags |
| `VARCHAR(x)` | 65,535 chars (row) | Variable-size, efficient | Names, titles, emails |
| `TINYTEXT` | 255 chars | Small text with 1-byte len | Tags, short blurbs |
| `TEXT` | 65,535 chars | Medium text, 2-byte len | Comments, descriptions |
| `MEDIUMTEXT` | ~16 million chars | Large text, 3-byte len | Logs, JSON, large notes |
| `LONGTEXT` | ~4 billion chars | Very large, 4-byte len | Books, HTML, full articles |

**INTEGER Data Types**

| Data Type | Storage | Signed Range | Unsigned Range |
| --- | --- | --- | --- |
| `TINYINT` | 1 byte | –128 to 127 | 0 to 255 |
| `SMALLINT` | 2 bytes | –32,768 to 32,767 | 0 to 65,535 |
| `MEDIUMINT` | 3 bytes | –8.38M to 8.38M | 0 to 16.77M |
| `INT` | 4 bytes | –2.14B to 2.14B | 0 to 4.29B |
| `BIGINT` | 8 bytes | –9.22Q to 9.22Q | 0 to 18.4Q |
- **ZEROFILL** pads the displayed number with leading zeros to fill the specified width.

```sql
CREATE TABLE items (item_id INT(5) ZEROFILL);
INSERT INTO items VALUES (42);
-- SELECT * FROM items; returns '00042'
```

**RATIONAL Data Types (Fixed-Point and Floating-Point)**

| Type | Precision | Exact? | Use Case | Storage |
| --- | --- | --- | --- | --- |
| `DECIMAL(p, s)` | Fixed-point | ✔ Yes | Money, totals | Variable |
| `FLOAT` | ~6–7 digits | X No | Scientific values | 4 bytes |
| `DOUBLE` | ~15–16 digits | X No | More precise calculations | 8 bytes |

**BOOLEANs**

- In MySQL, `BOOLEAN` is an alias for `TINYINT(1)`. `TRUE` is stored as `1` and `FALSE` is stored as `0`.

**ENUM and SET**

- These are string object types used to restrict column values to predefined lists.
- `ENUM`: A single value chosen from a list.
- `SET`: Multiple values can be selected from a list.

```sql
CREATE TABLE products (features SET('wifi', 'bluetooth', 'gps'));
```

**DATE/TIME Data Types**

| Type | Stores | Format | Time Zone Aware? | Auto-Update? | Size |
| --- | --- | --- | --- | --- | --- |
| `DATE` | Date only | `YYYY-MM-DD` | X No | X No | 3 Bytes |
| `TIME` | Time only | `HH:MM:SS` | X No | X No | 3 Bytes |
| `DATETIME` | Date & Time | `YYYY-MM-DD HH:MM:SS` | X No | X No | 8 Bytes |
| `TIMESTAMP` | Date & Time | `YYYY-MM-DD HH:MM:SS` | ✔ Yes | ✔ Yes (optional) | 4 Bytes |
| `YEAR` | Year only | `YYYY` | X No | X No | 1 Byte |

**BLOBs (Binary Large Objects)**

- Used to store binary data like images, audio, or PDFs. It's generally **not recommended** to store large files in a database due to performance, backup, and scalability issues. It's better to store the files on a file system or object store (like AWS S3) and save the file path or URL in the database.

**JSON**

- A native data type in modern SQL databases for storing semi-structured data.

```sql
-- Create a JSON value
UPDATE products
SET properties = '{
	"dimensions": [1, 2, 3],
	"weight": 10,
	"manufacturer": {"name": "sony"}
}'
WHERE product_id = 1;

-- Extract a value from JSON
SELECT product_id, JSON_EXTRACT(properties, '$.weight') AS weight
FROM products
WHERE product_id = 1;

-- Update a JSON value
UPDATE products
SET properties = JSON_SET(properties, '$.weight', 20, '$.age', 10)
WHERE product_id = 1;
```

### **5.3 Character Sets and Collations**

- A **character set** is the set of symbols and encodings (the "alphabet"). `utf8mb4` is recommended for full Unicode support.
- A **collation** is a set of rules for comparing and sorting characters (e.g., case-sensitive vs. case-insensitive). `utf8mb4_unicode_ci` is a common choice.

```sql
SHOW CHARSET;
-- Set character set and collation for a table
ALTER TABLE customers
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

### **5.4 Storage Engines**

A **storage engine** is the underlying software component that a database uses to create, read, update, and delete data. **InnoDB** is the default and recommended engine for most use cases as it is ACID-compliant and supports transactions and foreign keys.

| **Engine** | **Transactions** | **Foreign Keys** | **Locking** | **Notes** |
| --- | --- | --- | --- | --- |
| **InnoDB** | ✔ Yes | ✔ Yes | Row-level | Default engine; best for most applications. |
| **MyISAM** | X No | X No | Table-level | Fast for reads, but lacks modern features. |
| **MEMORY** | X No | X No | Table-level | Data stored in RAM; volatile. For temporary tables. |

# **Chapter VI: Performance and Optimization**

### **6.1 Indexing**

An **index** is a data structure that improves the speed of data retrieval operations on a database table at the cost of additional writes and storage space.

**Types of Indexes**

- **Single Column Index**: An index on a single column.
- **Composite Index**: An index on multiple columns. The order of columns is critical.
- **Prefix Index**: An index on the first `N` characters of a string column.
- **FULLTEXT Index**: Used for natural language search (`MATCH ... AGAINST`).

```sql
-- Create a simple index
CREATE INDEX idx_points ON customers (points);

-- Create a composite index
USE sql_store;
CREATE INDEX idx_state_points ON customers (state, points);

-- Query that uses the composite index
SELECT customer_id FROM customers WHERE state = 'CA' AND points > 1000;

-- Create a prefix index on the first 5 characters of last_name
CREATE INDEX idx_lastname ON customers (last_name(5));

-- Show and drop indexes
SHOW INDEXES IN customers;
DROP INDEX idx_state_points ON customers;
```

**Composite Index Column Order** When creating a composite index, the order of columns matters immensely.

| **Rule** | **Explanation** |
| --- | --- |
| **Put most selective columns first** | Columns with high cardinality (many unique values, like `user_id`) should come first to filter down the data faster. |
| **Match your queries** | The index should match the column order in your most common `WHERE`, `JOIN`, and `ORDER BY` clauses. An index on `(state, points)` can be used for queries on `state` or on `state AND points`, but not for queries only on `points`. |

### **6.2 Query Performance and Optimization**

Understanding *why* a query is slow is a critical skill. The `EXPLAIN` command is the primary tool for this.

- **`EXPLAIN`**: Shows the execution plan that the database optimizer chooses for a query. It reveals how tables are joined, what indexes are used, and potential performance bottlenecks.

```sql
EXPLAIN SELECT customer_id FROM customers WHERE state = 'CA' AND points > 1000;
```

**Key Columns in `EXPLAIN` Output**

- **`type`**
    - **What it Tells You:** The join type used.
    - **What to Look For:** `ALL` is a full table scan (bad). `index`, `range`, `ref`, `eq_ref`, and `const` are progressively better.
- **`possible_keys`**
    - **What it Tells You:** Indexes that the optimizer *could* use for this query.
    - **What to Look For:** If this is `NULL`, you may be missing a relevant index.
- **`key`**
    - **What it Tells You:** The index that was *actually* used.
    - **What to Look For:** If this is `NULL` but `possible_keys` is not, the optimizer chose not to use the index (the query might need rewriting).
- **`rows`**
    - **What it Tells You:** The estimated number of rows the database must scan.
    - **What to Look For:** A high number indicates an inefficient query.
- **`Extra`**
    - **What it Tells You:** Additional information about the query execution.
    - **What to Look For:** `Using filesort` means an expensive sort operation that couldn't use an index. `Using temporary` means a temporary table was created. Both are often signs of a sub-optimal query.

# **Chapter VII: Security and User Management**

### **7.1 Creating and Removing Accounts**

```sql
-- See existing users
SELECT user, host FROM mysql.user;

-- Create a new user identified by a password
CREATE USER 'alex'@'localhost' IDENTIFIED BY '1234';

-- Drop a user
DROP USER 'alex'@'localhost';
```

### **7.2 Granting and Revoking Privileges**

```sql
-- Grant specific privileges on a specific table to a user
GRANT SELECT, INSERT, UPDATE, DELETE, EXECUTE
	ON sql_store.customers
    TO 'alex'@'localhost';

-- Grant all privileges on all databases (admin-level)
GRANT ALL PRIVILEGES
    ON *.*
    TO 'john'@'localhost' WITH GRANT OPTION;

-- See the grants for a specific user
SHOW GRANTS FOR 'john'@'localhost';

-- Revoke a specific privilege
REVOKE CREATE VIEW
    ON sql_store.*
    FROM 'alex'@'localhost';
```