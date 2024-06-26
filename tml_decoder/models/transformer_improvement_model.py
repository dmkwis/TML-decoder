from ast import literal_eval
import os
import neptune
from openai import OpenAI
import pandas as pd
from tml_decoder.models.abstract_model import AbstractLabelModel


class TransformerImprovementModel(AbstractLabelModel):
        @staticmethod
        def _get_run_data(id: str):
                project = neptune.init_project(
                        project=os.getenv("NEPTUNE_PROJECT"),
                        api_token=os.getenv("NEPTUNE_API_TOKEN"),
                        mode="read-only"
                )

                df = project.fetch_runs_table(id=[id]).to_pandas()

                assert len(df) == 1

                generated_labels = df.filter(regex='predictions/test/.*/generated_label').iloc[0].tolist()
                true_labels = df.filter(regex='predictions/test/.*/category').iloc[0].tolist()

                print(len(true_labels))

                return generated_labels, true_labels
        
        @staticmethod
        def _get_summaries(path: str):
                return pd.read_csv(path, converters={'subs': literal_eval})

        def __init__(self, original_run_id: str, summaries_path: str):
                super().__init__()
                
                self.original_run_id = original_run_id
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                generated_labels, true_labels = TransformerImprovementModel._get_run_data(original_run_id)
                self.summaries_df = TransformerImprovementModel._get_summaries(summaries_path)
                self.texts_to_generated_labels = []
                # for _, row in self.summaries_df.iterrows():
                #         if row['description'] not in true_labels:
                #                 continue
                #         else:
                #                 self.texts_to_generated_labels.append((row['subs'], generated_labels[true_labels.index(row['description'])]))
                for true_label in true_labels:
                        self.texts_to_generated_labels.append((self.summaries_df[self.summaries_df['description'] == true_label]['subs'].values[0], generated_labels[true_labels.index(true_label)]))

        @property
        def name(self):
                return f"Transformer improvement model, run id: {self.original_run_id}"
        

        def _get_improved_prediction(self, prediction: str) -> str:
                """
                Improve a prediction using the OpenAI API.

                :param client: An instance of the OpenAI class.
                :param prediction: The prediction to improve.
                """

                response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                                {
                                "role": "system",
                                "content": """
                                        You are dealing with a machine generated summary of a data cluster.
                                        The summary is not perfect and you need to improve it. Please provide a
                                        more accurate and concise summary of the data cluster. The data cluster
                                        is a collection of data points that are related to each other. The summary
                                        should be a couple of words long and should capture the key insights from 
                                        the data cluster. Respond ONLY with the improved summary of the data cluster. 
                                        Remember to be very concise and capture the key insights. 
                                        It has to be descriptive and human friendly.""",
                                },
                                {"role": "user", "content": prediction},
                        ]
                        
                )

                response = response.choices[0].message.content

                return response if response else ""


        def get_label(self, texts: list[str]) -> str:
                for subs, generated_label in self.texts_to_generated_labels:
                        # if texts are a subset of subs
                        if all(text in subs for text in texts):
                                improved_label = self._get_improved_prediction(generated_label)
                                print(f"Original label: {generated_label}, improved label: {improved_label}")
                                return improved_label
                        
                print(texts, "ERROR SKURWYSYN")
                raise ValueError("No label found for the given texts")
