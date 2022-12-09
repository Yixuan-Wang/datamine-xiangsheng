from typing import Optional

import torch
import torch.nn
from transformers import BertModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
)


class ModelMultipleChoice(torch.nn.Module):
    def __init__(self, bert: BertModel) -> None:
        super().__init__()

        self.bert = bert
        classifier_dropout: float = (
            self.bert.config.classifier_dropout
            if self.bert.config.classifier_dropout is not None
            else self.bert.config.hidden_dropout_prob
        )
        self.dropout = torch.nn.Dropout(classifier_dropout)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 1)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> MultipleChoiceModelOutput:

        # ? input shape: [n, c, b]
        # ?   n: input tensor length
        # ?   c: number of choices
        # ?   b: batch size

        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = self.dropout(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        loss = self.criterion(reshaped_logits, labels) if labels is not None else None

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ModelNextSentencePrediction(torch.nn.Module):
    def __init__(self, bert: BertModel):
        super().__init__()

        self.bert = bert
        self.seq_relationship = torch.nn.Linear(self.bert.config.hidden_size, 2)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs.pooler_output
        seq_relationship_scores = self.seq_relationship(pooled_output)

        loss = (
            self.criterion(seq_relationship_scores.view(-1, 2), labels.view(-1))
            if labels is not None
            else None
        )

        return NextSentencePredictorOutput(
            loss=loss,
            logits=seq_relationship_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["ModelMultipleChoice", "ModelNextSentencePrediction"]
