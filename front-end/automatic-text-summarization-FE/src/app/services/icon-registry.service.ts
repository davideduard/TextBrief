import { Injectable } from '@angular/core';
import { MatIconRegistry } from '@angular/material/icon';
import { DomSanitizer } from '@angular/platform-browser';

@Injectable({
	providedIn: 'root'
})
export class IconRegistryService {
	constructor(
		private matIconRegistry: MatIconRegistry,
		private domSanitizer: DomSanitizer
	) {}

	registerCustomIcons(): void {
		this.matIconRegistry.addSvgIcon(
			'arrow',
			this.domSanitizer.bypassSecurityTrustResourceUrl(
				'/assets/icons/arrow.svg'
			)
		);

		this.matIconRegistry.addSvgIcon(
			'microphone',
			this.domSanitizer.bypassSecurityTrustResourceUrl(
				'/assets/icons/microphone.svg'
			)
		);

		this.matIconRegistry.addSvgIcon(
			'microphone-off',
			this.domSanitizer.bypassSecurityTrustResourceUrl(
				'/assets/icons/microphone-off.svg'
			)
		);
	}
}
