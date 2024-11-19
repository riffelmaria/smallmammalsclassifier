import pandas as pd
from opensoundscape import load_model
from opensoundscape.utils import make_clip_df
from shared.audio_functions import walk_wav
from shared.files_manipulation_functions import find_files


def create_audio_df(folder: str, prediction_steps: float) -> pd.DataFrame:
    audiofiles = find_files(folder, walk_wav)
    df = make_clip_df(
        files=audiofiles,
        clip_duration=prediction_steps,
        clip_overlap=0,
        final_clip=None,
        return_invalid_samples=False,
        raise_exceptions=False,
    )

    return df  # type: ignore # TODO Koni


def prediction(
    model_path: str,
    df: pd.DataFrame,
    activation_layer: str = "sigmoid",
    workers: int = 0,
    wandb_session: str | None = None,
    split_files: bool = True,  # if one prediction per clip and not per file is needed
):
    model = load_model(model_path)
    prediction_scores_df = model.predict(
        df,
        activation_layer=activation_layer,  # 'sigmoid' transforms each score individually to [0,1] without requiring that all scores sum to 1
        num_workers=workers,
        wandb_session=wandb_session,
        final_clip="full",
        split_files_into_clips=split_files,
    )
    return prediction_scores_df


def round_by_threshold(x, threshold):
    return 1 if x >= threshold else 0


def use_decision_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # round the probability column
    classified_df = df.copy()
    # Loop over columns and apply the rounding function
    for column in classified_df.columns:
        classified_df[column] = classified_df[column].apply(
            lambda x: round_by_threshold(x, threshold)
        )
    # Add threshold as a new column
    classified_df["threshold"] = threshold

    return classified_df


# round_all_classes <- function(pred_df, threshold = 0.7){

#     classes <- names(pred_df)[-c(1,2,3)]

#     for (class in classes){
#         new_values <- c(round_threshold(pred_df[[class]], threshold = threshold))

#         if (length(new_values) == nrow(pred_df)) {
#             pred_df[[class]] <- new_values
#         } else {
#             stop("Length of new_values must be equal to the number of rows in the data frame.")
#         }
#     }
#     return(pred_df)
# }


# aggregate_classes <- function(df, threshold){ # bearbeiten!
#     classes_map <- list(
#         "bats" = c("Eptesicus","Myotis","Nyctalus","Plecotus", "Pipistrellus"),
#         "targetMammal" = c("apodemus","Mus","Micromys","Arvicola","Microtus","Sorex","Neomys","Myodes","rattus", "Crocidura"),
#         "Noise" = c("Noise")
#     )
#     for (key in names(classes_map)) {
#         # Extract column names for rowMeans calculation
#         cols_to_mean <- classes_map[[key]]

#         # Round each value depending on a given threshold
#         df[cols_to_mean] <- lapply(df[cols_to_mean], round_threshold, threshold = threshold)

#         # Calculate rowMeans based on selected columns
#         # Compute the sum of each row
#         row_sums <- rowSums(df[cols_to_mean])

#         # Cap the sum at 1 if it exceeds 1
#         capped_sums <- pmin(row_sums, 1)
#         df[[key]] <- capped_sums

#         #df[[key]] <- rowMeans(df[cols_to_mean])
#     }

#     df <- df %>% select(all_of(c("file", "start_time", "end_time", "targetMammal", "bats", "Noise")))
#     return(df)
# }
