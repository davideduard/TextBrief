import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
	selector: 'app-home-page-component',
	template: `
		<div class="w-full h-full flex flex-col items-center">
			<div class="w-full h-16 bg-green flex items-center pl-14">
				<p class="text-white font-bold">Summary Generator</p>
			</div>
			<div class="flex flex-col items-center gap-5 mt-10">
				<p class="text-black font-bold text-2xl">Representation Used</p>
				<div class="flex flex-row gap-16">
					<mat-button-toggle-group
						value="embeddings"
						(change)="onRepresentationChange($event)"
					>
						<mat-button-toggle value="embeddings" class="w-52"
							>Word Embeddings</mat-button-toggle
						>
						<mat-button-toggle value="tfidf" class="w-52"
							>TF-IDF</mat-button-toggle
						>
						<mat-button-toggle value="jaccard" class="w-52"
							>Bag of Words</mat-button-toggle
						>
					</mat-button-toggle-group>
				</div>
				<div class="flex flex-row gap-48 relative">
					<div class="flex flex-col gap-5 items-center mt-10">
						<p class="text-4xl font-bold">Input Text</p>
						<div class="bg-gray w-96 h-[450px] relative">
							<textarea
								class="w-[93%] h-full bg-gray overflow-scroll no-scrollbar resize-none p-4 focus:outline-none text-justify"
								placeholder="Enter a text..."
								[ngClass]="{ 'placeholder-red outline-red': hasError }"
								[(ngModel)]="inputText"
							>
							</textarea>
							<button
								class="absolute bottom-2 right-3 w-7 h-7 scale-[1.2]"
								(click)="onSpeechButtonClick()"
							>
								<mat-icon *ngIf="!isListening" svgIcon="microphone"></mat-icon>
								<mat-icon
									*ngIf="isListening"
									svgIcon="microphone-off"
								></mat-icon>
							</button>
						</div>
					</div>

					<div class="flex flex-col gap-5 items-center mt-10">
						<p class="text-4xl font-bold">Summary</p>
						<div class="bg-gray w-96 h-[450px]">
							<textarea
								class="w-full h-full bg-gray overflow-scroll no-scrollbar resize-none p-4 focus:outline-none hover:cursor-default"
								readonly
								placeholder="This is where the generated text will appear"
								[value]="generatedSummary"
							></textarea>
						</div>
					</div>

					<mat-icon
						svgIcon="arrow"
						class="scale-[8] absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
					></mat-icon>
				</div>
				<div class="mt-16 w-36">
					<app-flat-button
						label="Generate"
						(click)="onClick()"
						[isLoading]="isLoading"
					></app-flat-button>
				</div>
			</div>
		</div>
	`,
	styleUrls: ['./home-page.component.scss']
})
export class HomePageComponent {
	hasError: boolean = false;
	selectedRepresentation: string = 'embeddings';

	@Input() generatedSummary: string = '';
	@Input() isLoading: boolean = false;
	@Input() inputText: string = '';
	@Input() isListening: boolean = false;

	@Output() inputTextEmit: EventEmitter<string> = new EventEmitter<string>();
	@Output() representationEmit: EventEmitter<string> =
		new EventEmitter<string>();
	@Output() speechButtonEmit: EventEmitter<void> = new EventEmitter<void>();

	onClick(): void {
		if (this.inputText != '') {
			this.hasError = false;
			this.inputTextEmit.emit(this.inputText);
		} else {
			this.hasError = true;
		}
	}

	onRepresentationChange(event: any): void {
		this.selectedRepresentation = event.value;
		this.representationEmit.emit(this.selectedRepresentation);
	}

	onSpeechButtonClick(): void {
		this.speechButtonEmit.emit();
	}
}
