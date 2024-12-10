import csv
import matplotlib.pyplot as plt
import numpy as np

def parse_accuracy(value):
    """Extracts the accuracy from a string representing a tuple."""
    try:
        # Convert the string to a tuple and extract the second value (accuracy)
        return float(value.strip("()").split(",")[1].strip())
    except Exception as e:
        print(f"Error parsing accuracy value '{value}': {e}")
        return None

def save_accuracy_plot(file_path, output_image_path):
    # Read CSV file
    data = {}
    with open(file_path, mode="r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            attacker_percentage = int(row["Attacker Percentage"])
            round_number = int(row["Round"])
            accuracy = parse_accuracy(row["Accuracy"])  # Parse the accuracy value
            
            if accuracy is not None:  # Skip rows with invalid accuracy
                if attacker_percentage not in data:
                    data[attacker_percentage] = []
                data[attacker_percentage].append((round_number, accuracy))
    
    # Plotting
    plt.figure(figsize=(12, 8))  # Increase the figure size for better readability
    for attacker_percentage, values in sorted(data.items()):
        rounds, accuracies = zip(*values)  # Unzip rounds and accuracies
        plt.plot(
            rounds, 
            accuracies, 
            marker="o", 
            markersize=6,  # Increase marker size for visibility
            linewidth=2,  # Increase line width for clarity
            label=f"{attacker_percentage}% Attackers"
        )

    # Graph formatting
    plt.title(
        "Federated Learning Accuracy by Round for Different Attacker Percentages", 
        fontsize=16, 
        weight="bold"
    )
    plt.xlabel("Round Number", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.xticks(np.arange(1, 11, 1), fontsize=12)
    plt.yticks(np.arange(70, 95, 1), fontsize=12)  # Y-axis ticks with 1% increments
    plt.legend(title="Attacker Percentage", fontsize=12, title_fontsize=14)
    plt.grid(which="major", color="gray", linestyle="--", linewidth=0.5)  # Major gridlines
    plt.grid(which="minor", color="gray", linestyle=":", linewidth=0.3)  # Minor gridlines
    plt.minorticks_on()  # Turn on minor ticks
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig(output_image_path)
    plt.close()  # Close the plot to free up memory

# Example usage
csv_file_path = "roundwise_accuracy.csv"
output_image_path = "accuracy_by_round_detailed.png"
save_accuracy_plot(csv_file_path, output_image_path)
