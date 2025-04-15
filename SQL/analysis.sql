/* 
================================== Performance Analysis =====================================
*/

-- Total Revenue by Year

USE AdventureWorksDW2022;

SELECT d.CalendarYear, 
       SUM(f.SalesAmount) AS Total_Revenue
FROM FactInternetSales f
JOIN DimDate d ON f.OrderDateKey = d.DateKey
GROUP BY d.CalendarYear
ORDER BY d.CalendarYear;

-- Top 10 best selling products
SELECT
	TOP 10 p.EnglishProductName,
	SUM(s.SalesAmount) AS TotalAmountSold,
	SUM(s.OrderQuantity) AS TotalUnitsSold
FROM DimProduct p
JOIN FactInternetSales s
ON p.productkey = s.ProductKey
GROUP BY p.EnglishProductName
ORDER BY TotalAmountSold DESC;

-- Monthly Sales Growth Rate, Month-over-Month(MoM)

WITH MonthlySales AS(
	SELECT
		d.CalendarYear AS [year],
		d.MonthNumberOfYear As [month],
		SUM(s.SalesAmount) As total_sales 
	FROM 
		FactInternetSales s
	JOIN
		DimDate d
	ON d.DateKey = s.OrderDateKey
	GROUP BY d.CalendarYear, d.MonthNumberOfYear
)
SELECT 
    [year], 
    [month], 
    Total_Sales, 
    LAG(total_sales) OVER (ORDER BY [year], [month]) AS Previous_Month_Sales,
    ((total_sales - LAG(total_sales) OVER (ORDER BY [year], [month])) / 
     LAG(total_sales) OVER (ORDER BY [year], [month])) * 100 AS MoM_Growth_Rate,
	 CASE
		WHEN ((total_sales - LAG(total_sales) OVER (ORDER BY [year], [month])) / 
     LAG(total_sales) OVER (ORDER BY [year], [month])) * 100 > 0 THEN 'positive growth'
		WHEN ((total_sales - LAG(total_sales) OVER (ORDER BY [year], [month])) / 
     LAG(total_sales) OVER (ORDER BY [year], [month])) * 100 = 0 THEN 'no growth'
		WHEN ((total_sales - LAG(total_sales) OVER (ORDER BY [year], [month])) / 
     LAG(total_sales) OVER (ORDER BY [year], [month])) * 100 < 0 THEN 'negative growth'
	END growth
FROM MonthlySales;

-- Average Order Value (AoV)

SELECT 
	SUM(SalesAmount) / COUNT(DISTINCT SalesOrderNumber) AS Average_Order_Value
FROM 
	FactInternetSales;

-- Revenue by Employee

SELECT
	e.EmployeeKey,
	e.FirstName,
	e.LastName,
	SUM(s.SalesAmount) AS total_sales
FROM 
	FactResellerSales s
JOIN
	DimEmployee e
ON s.EmployeeKey = s.EmployeeKey
GROUP BY e.EmployeeKey, e.FirstName, e.LastName;

/* 
================================== Customer Analysis =====================================
*/

-- Customer lifetime value
SELECT 
	c.CustomerKey,
	c.FirstName,
	c.LastName,
	SUM(s.SalesAmount) AS lifetime_value
FROM 
	FactInternetSales s
JOIN
	DimCustomer c
ON c.CustomerKey = s.CustomerKey
GROUP BY 
	c.CustomerKey, c.FirstName, c.LastName
ORDER BY lifetime_value DESC;

-- Customer retention rate
WITH order_count AS(
SELECT
	s.CustomerKey,
	COUNT(DISTINCT SalesOrderNumber) AS orderCount
FROM 
	FactInternetSales s
GROUP BY 
	s.CustomerKey)

SELECT
	FORMAT((COUNT(CASE WHEN orderCount > 1 THEN CustomerKey END) * 100.0) / COUNT(CustomerKey),'N2') AS Retention_Rate
FROM order_count;

-- Recency, Frequency, Monetary value

SELECT
	c.CustomerKey,
	c.FirstName,
	c.LastName,
	SUM(s.SalesAmount) AS Monetary_value,
	MAX(s.OrderDate) AS Recency,
	COUNT(s.SalesOrderNumber) AS Frequency
FROM
	FactInternetSales s
JOIN 
	DimCustomer c
ON c.CustomerKey = s.CustomerKey
GROUP BY c.CustomerKey, c.FirstName, c.LastName;

-- Customer geographic distribution

SELECT
	g.EnglishCountryRegionName,
	COUNT(DISTINCT c.CustomerKey) AS total_customers
FROM
	DimCustomer c
JOIN
	DimGeography g
ON g.GeographyKey = c.GeographyKey
GROUP BY g.EnglishCountryRegionName;

/* 
================================== Inventory Analysis =====================================
*/

-- Inventory turn_over ratio

WITH InventoryValue AS (
    SELECT 
        i.ProductKey,
        AVG(i.UnitsBalance) AS Avg_Stock, 
        p.StandardCost
    FROM FactProductInventory i
    JOIN DimProduct p ON i.ProductKey = p.ProductKey
    GROUP BY i.ProductKey, p.StandardCost
)
SELECT 
    SUM(f.TotalProductCost) / NULLIF(SUM(iv.Avg_Stock * iv.StandardCost), 0) AS Inventory_Turnover_Ratio
FROM FactInternetSales f
JOIN InventoryValue iv 
ON f.ProductKey = iv.ProductKey;

-- Stock Availability

WITH Latest_Stock AS (
    SELECT 
        i.ProductKey,
        p.EnglishProductName AS ProductName,
        AVG(i.UnitsBalance) AS Avg_Current_Stock  -- Use average to avoid overstatement
    FROM FactProductInventory i
    JOIN DimProduct p ON i.ProductKey = p.ProductKey
    WHERE i.DateKey = (SELECT MAX(DateKey) FROM FactProductInventory)  -- Get latest stock levels
    GROUP BY i.ProductKey, p.EnglishProductName
)
SELECT 
    ls.ProductName,
    ls.Avg_Current_Stock, 
    SUM(s.OrderQuantity) AS Total_Sales,
    (ls.Avg_Current_Stock * 1.0) / NULLIF(SUM(s.OrderQuantity), 0) AS Stock_to_Sales_Ratio,
	CASE 
		WHEN (ls.Avg_Current_Stock * 1.0) / NULLIF(SUM(s.OrderQuantity), 0) < 0.5 THEN 'Understocked'
		WHEN (ls.Avg_Current_Stock * 1.0) / NULLIF(SUM(s.OrderQuantity), 0) BETWEEN 0.5 AND 2.0 THEN 'Adequate'
		ELSE 'Overstocked'
		END AS stockingLevel
FROM Latest_Stock ls
JOIN FactInternetSales s ON ls.ProductKey = s.ProductKey
GROUP BY ls.ProductName, ls.Avg_Current_Stock
HAVING SUM(s.OrderQuantity) > 5;  -- Filter out products with very low sales to avoid extreme ratios

-- Days of inventory on Hand(DOH)

WITH Latest_Stock AS (
    SELECT 
        i.ProductKey,
        p.EnglishProductName AS ProductName,
        AVG(i.UnitsBalance) AS Avg_Current_Stock  -- Use average to smooth fluctuations
    FROM FactProductInventory i
    JOIN DimProduct p ON i.ProductKey = p.ProductKey
    WHERE i.DateKey = (SELECT MAX(DateKey) FROM FactProductInventory)  -- Only latest stock levels
    GROUP BY i.ProductKey, p.EnglishProductName
)
SELECT 
    ls.ProductName,
    ls.Avg_Current_Stock, 
    SUM(f.OrderQuantity) AS Total_Sales,
    CASE 
        WHEN (ls.Avg_Current_Stock * 365.0) / NULLIF(SUM(f.OrderQuantity), 0) > 365 
        THEN FORMAT(365, 'N2')  -- Capping at 1 year max
        ELSE FORMAT((ls.Avg_Current_Stock * 365.0) / NULLIF(SUM(f.OrderQuantity), 0),'N2') 
    END AS Days_of_Inventory_On_Hand
FROM Latest_Stock ls
JOIN FactInternetSales f ON ls.ProductKey = f.ProductKey
GROUP BY ls.ProductName, ls.Avg_Current_Stock
HAVING SUM(f.OrderQuantity) > 10  -- Removing low-sales products
ORDER BY Days_of_Inventory_On_Hand DESC;

-- Backorder Date

SELECT 
    FORMAT((COUNT(CASE WHEN i.UnitsBalance = 0 THEN f.SalesOrderNumber END) * 100.0) / COUNT(f.SalesOrderNumber), 'N2') AS Backorder_Rate
FROM FactProductInventory i
JOIN FactInternetSales f ON i.ProductKey = f.ProductKey;

/* 
================================== Profitability Analysis =====================================
*/

-- Gross profit margin

SELECT 
    p.EnglishProductName AS ProductName, 
    SUM(s.SalesAmount - s.TotalProductCost) AS Gross_Profit,
    (SUM(s.SalesAmount - s.TotalProductCost) / NULLIF(SUM(s.SalesAmount), 0)) * 100 AS Gross_Profit_Margin
FROM FactInternetSales s
JOIN DimProduct p ON s.ProductKey = p.ProductKey
GROUP BY p.EnglishProductName
ORDER BY Gross_Profit_Margin DESC;

-- Profit by region

SELECT 
    t.SalesTerritoryRegion, 
    SUM(s.SalesAmount - s.TotalProductCost) AS Total_Profit
FROM FactInternetSales s
JOIN DimSalesTerritory t ON s.SalesTerritoryKey = t.SalesTerritoryKey
GROUP BY t.SalesTerritoryRegion
ORDER BY Total_Profit DESC;

-- Customer lifetime Value (CLV)

SELECT 
    c.CustomerKey, 
    c.FirstName, 
    c.LastName, 
    SUM(s.SalesAmount) AS Customer_Lifetime_Value
FROM FactInternetSales s
JOIN DimCustomer c ON s.CustomerKey = c.CustomerKey
GROUP BY c.CustomerKey, c.FirstName, c.LastName
ORDER BY Customer_Lifetime_Value DESC;

-- Customer churn rate

WITH LastPurchase AS (
    SELECT 
        CustomerKey, 
        MAX(OrderDateKey) AS Last_Purchase_Date
    FROM FactInternetSales
    GROUP BY CustomerKey
)
SELECT 
    (COUNT(CASE WHEN DATEDIFF(DAY, d.FullDateAlternateKey, GETDATE()) > 365 THEN CustomerKey END) * 100.0) 
    / COUNT(CustomerKey) AS Churn_Rate
FROM LastPurchase lp
JOIN DimDate d ON lp.Last_Purchase_Date = d.DateKey;

/* 
================================== Employee Analysis =====================================
*/

-- Employee performance by revenue

SELECT 
    e.EmployeeKey, 
    e.FirstName, 
    e.LastName, 
    SUM(s.SalesAmount) AS Total_Sales
FROM FactResellerSales s
JOIN DimEmployee e ON s.EmployeeKey = e.EmployeeKey
GROUP BY e.EmployeeKey, e.FirstName, e.LastName
ORDER BY Total_Sales DESC;

/* 
================================== Marketing Analysis =====================================
*/

-- Sales based on discounts

SELECT 
    d.EnglishPromotionName, 
    SUM(f.SalesAmount) AS Total_Revenue
FROM FactInternetSales f
JOIN DimPromotion d ON f.PromotionKey = d.PromotionKey
GROUP BY d.EnglishPromotionName
ORDER BY Total_Revenue DESC;


