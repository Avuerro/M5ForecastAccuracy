import pandas as pd


def create_submission_csv(predictions,submission_file, filename):
  # create submission file
  predictions_copy = predictions
  final_submission = pd.concat([predictions, predictions_copy])
  final_submission.reset_index(drop=True, inplace=True)
  final_submission = final_submission.astype(int)
  final_submission.insert(0, 'id', submission_file['id'])
  final_submission.columns = ['id'] + [f"F{i}" for i in range(1, 29)]

  final_submission.to_csv('{}.csv'.format(filename), index=False)
