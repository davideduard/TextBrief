import { Injectable } from '@angular/core';

declare global {
	interface IWindow {
		webkitSpeechRecognition: any;
	}
}

@Injectable({
	providedIn: 'root'
})
export class SpeechRecognitionService {
	recognition: any;
	isListening = false;
	private transcript: string = '';

	constructor() {
		const { webkitSpeechRecognition }: IWindow = <IWindow>(<unknown>window);
		this.recognition = new webkitSpeechRecognition();
		this.recognition.lang = 'ro-RO';
		this.recognition.continuous = false;
		this.recognition.interimResults = true;

		this.recognition.onresult = (event: any) => {
			this.transcript = '';
			for (let i = event.resultIndex; i < event.results.length; i++) {
				this.transcript += event.results[i][0].transcript;
			}
		};

		this.recognition.onend = () => {
			this.isListening = false;
		};

		this.recognition.onerror = (event: any) => {
			console.error(event.error);
			this.stopListening();
		};
	}

	startListening(callback: (transcript: string) => void): void {
		if (this.isListening) {
			return;
		}

		this.isListening = true;
		this.recognition.start();

		this.recognition.onresult = (event: any) => {
			this.transcript = '';
			for (let i = event.resultIndex; i < event.results.length; i++) {
				this.transcript += event.results[i][0].transcript;
			}
			callback(this.transcript);
		};
	}

	stopListening(): void {
		if (!this.isListening) {
			return;
		}

		this.isListening = false;
		this.recognition.stop();
	}
}
