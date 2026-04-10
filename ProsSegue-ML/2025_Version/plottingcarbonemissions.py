import matplotlib.pyplot as plt
import pandas as pd


# Plotting here
df = pd.read_csv('emissions.csv')

df['project_name'] = df['project_name'].str.extract(r'^(\w+)')

grouped = df.groupby('project_name')['emissions'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
grouped.plot(kind='bar', color='green')
plt.title('CO₂ Emissions by ML Model')
plt.ylabel('Emissions (kg CO₂)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

grouped = df.groupby('project_name')['emissions_rate'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
grouped.plot(kind='bar', color='green')
plt.title('CO₂ Emissions Rate by ML Model')
plt.ylabel('Emissions (kg CO₂/s)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

grouped = df.groupby('project_name')['duration'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
grouped.plot(kind='bar', color='green')
plt.title('Duration by ML Model')
plt.ylabel('Duration (?)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

grouped = df.groupby('project_name')['emissions'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
grouped.plot(kind='bar', color='green')
plt.title('Energy consumed by ML Model')
plt.ylabel('Energy consumed (kWh?)')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Precisa filtrar melhor aqui, pra plotar um gráfico duplo, sugiro gerar csv com uma seed só e filtrar o codecarbon fora, daí a gente gera mais certinho

# Create the dual-axis plot
#fig, ax1 = plt.subplots(figsize=(10, 6))

#color1 = 'tab:blue'
#ax1.set_xlabel('Model')
#ax1.set_ylabel('Total Emissions (kg CO₂)', color=color1)
#ax1.bar(models, emissions, color=color1, alpha=0.6, label='Emissions (kg CO₂)')
#ax1.tick_params(axis='y', labelcolor=color1)
#ax1.set_xticklabels(models, rotation=45)

# Create the second axis for emissions rate
#ax2 = ax1.twinx()
#color2 = 'tab:red'
#ax2.set_ylabel('Emissions Rate (kg CO₂/sec)', color=color2)
#ax2.plot(models, emissions_rate, color=color2, marker='o', label='Emissions Rate')
#ax2.tick_params(axis='y', labelcolor=color2)

#fig.tight_layout()
#plt.title('Total Emissions vs Emissions Rate by Model')
#plt.grid(True, which='both', linestyle='--', alpha=0.3)
#plt.show()

# test if numbers are small
#plt.yscale('log')
# to show in mg
#grouped_mg = grouped * 1e6
#grouped_mg.plot(kind='bar')
#plt.ylabel('Emissions (mg CO₂)')


