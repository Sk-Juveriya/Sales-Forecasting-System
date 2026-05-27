import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================
# 1. CREATE SAMPLE DATA
# =========================

np.random.seed(42)

days = np.arange(1, 61)  # 60 days

# Simulated sales trend (increasing + noise)
sales = days * 10 + np.random.randint(0, 50, size=len(days))

df = pd.DataFrame({
    "Day": days,
    "Sales": sales
})

print(df.head())

# =========================
# 2. TRAIN MODEL
# =========================

X = df[["Day"]]
y = df["Sales"]

model = LinearRegression()
model.fit(X, y)

# =========================
# 3. PREDICT FUTURE SALES
# =========================

future_days = np.arange(61, 71).reshape(-1, 1)
future_sales = model.predict(future_days)

# =========================
# 4. VISUALIZATION
# =========================

plt.figure(figsize=(8,5))

# Actual data
plt.scatter(df["Day"], df["Sales"], color="blue", label="Actual Sales")

# Best fit line
plt.plot(df["Day"], model.predict(X), color="green", label="Trend Line")

# Future prediction
plt.plot(future_days, future_sales, color="red", marker="o", label="Future Prediction")

plt.title("📈 Sales Forecasting System")
plt.xlabel("Day")
plt.ylabel("Sales")
plt.legend()
plt.show()

# =========================
# 5. PRINT FUTURE VALUES
# =========================

print("\n📊 Future Sales Prediction:")
for d, s in zip(future_days.flatten(), future_sales):
    print(f"Day {d} → Predicted Sales: {int(s)}")