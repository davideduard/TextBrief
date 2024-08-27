import { Injectable } from '@angular/core';
import { SummarizationRepository } from '../repositories';
import { Observable } from 'rxjs';
import { Summary } from '../types';

@Injectable({
	providedIn: 'root'
})
export class SummarizationService {
	constructor(private summarizationRepository: SummarizationRepository) {}

	requestSummary(text: string): Observable<Summary> {
		return this.summarizationRepository.requestSummary(text);
	}
}
