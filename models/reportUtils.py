import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

TEST = 100

def plot_metrics(history):
  metrics = ['loss', 'accuracy', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color='blue', label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color='blue', linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()
    
def generate_report(model, ds, y_values, model_name, ALL_LABELS, extra=""):
    report= classification_report(y_values, model.predict(ds).round(), target_names=ALL_LABELS, output_dict=True)
    df = pd.DataFrame(report).transpose()
    if extra:
        df.to_csv(f"{model_name}_report_{extra}.csv")
        df.to_latex(f"{model_name}_report_{extra}.tex")
    else:
        df.to_csv(f"{model_name}_report.csv")
        df.to_latex(f"{model_name}_report.tex")
    return df