import { Component } from '@angular/core';
import { SpeechRecognitionService } from '../../../../services';
import { SummarizationService } from '../../services';
import { Summary } from '../../types';

@Component({
	selector: 'app-home-page',
	template: `
		<div>
			<app-home-page-component
				(inputTextEmit)="generateSummary($event)"
				(representationEmit)="onRepresentationChange($event)"
				(speechButtonEmit)="toggleListening()"
				[generatedSummary]="generatedSummary"
				[inputText]="inputText"
				[isLoading]="isLoading"
				[isListening]="isListening"
			>
			</app-home-page-component>
		</div>
	`,
	styleUrls: ['./home-page.container.scss']
})
export class HomePageContainer {
	summaryResponse: Summary = { embeddings: '', jaccard: '', tfidf: '' };
	generatedSummary: string = '';
	selectedRepresentation: string = 'embeddings';
	isLoading: boolean = false;
	transcript: string = '';
	isListening: boolean = false;
	inputText: string = '';

	constructor(
		private speechRecognitionService: SpeechRecognitionService,
		private summarizationSerivce: SummarizationService
	) {}

	generateSummary(inputText: string): void {
		this.isLoading = true;

		this.summarizationSerivce
			.requestSummary(inputText)
			.subscribe((response: Summary) => {
				this.summaryResponse = response;
				console.log(response);
				this.generatedSummary =
					this.summaryResponse[this.selectedRepresentation as keyof Summary];
				this.isLoading = false;
			});
	}

	onRepresentationChange(selectedRepresentation: string): void {
		this.selectedRepresentation = selectedRepresentation;
		this.generatedSummary = '';
		this.isLoading = true;
		setTimeout(() => {
			this.isLoading = false;
			this.generatedSummary =
				this.summaryResponse[selectedRepresentation as keyof Summary];
		}, 1000);
	}

	toggleListening(): void {
		if (this.isListening) {
			this.speechRecognitionService.stopListening();
		} else {
			this.speechRecognitionService.startListening((transcript: string) => {
				this.inputText = transcript;
			});
		}
		this.isListening = !this.isListening;
	}
}
