import matplotlib.pyplot as plt

# Data from your results
methods = ["JL-5", "JL-15", "JL-20", "JL-25", "Original"]
times = [4.5, 12.2, 17.1, 19.8, 24.8]   # training times in seconds

plt.figure(figsize=(8,5))
bars = plt.bar(methods, times, color=['#4C72B0','#55A868','#C44E52','#8172B2','#CCB974'])

# Add labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{yval:.2f}s",
             ha='center', va='bottom', fontsize=10)

plt.title("Training Time Comparison: JL Projection vs Original", fontsize=14)
plt.ylabel("Training Time (seconds)")
plt.xlabel("Method")
plt.tight_layout()
plt.show()