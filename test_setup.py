import pathway as pw
import pandas as pd

# 1. Create a tiny fake dataset
data = {"claim": ["The sky is green", "The ocean is blue"], "id": [1, 2]}
df = pd.DataFrame(data)

# 2. Load it into Pathway
table = pw.debug.table_from_pandas(df)

# 3. Simple operation: Filter
result = table.filter(pw.this.claim == "The ocean is blue")

# 4. Run and print
pw.io.csv.write(result, "debug_output.csv")
pw.run()
print("Success! check debug_output.csv")