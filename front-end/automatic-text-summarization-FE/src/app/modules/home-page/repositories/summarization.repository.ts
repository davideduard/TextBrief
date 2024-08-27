import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { Summary, SummaryRequest } from '../types';

@Injectable({
	providedIn: 'root'
})
export class SummarizationRepository {
	private apiUrl = 'http://127.0.0.1:5000';
	constructor(private httpClient: HttpClient) {}

	requestSummary(text: string): Observable<Summary> {
		const requestData: SummaryRequest = { text };
		return this.httpClient.post<Summary>(
			`${this.apiUrl}/summarize`,
			requestData
		);
	}
}
