import { APP_INITIALIZER, NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HomePageModule } from './modules/home-page/home-page.module';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { ToggleGroupComponent } from './shared/components';
import { ToggleButtonComponent } from './shared/components';
import { IconRegistryService } from './services';
import { HttpClientModule } from '@angular/common/http';

@NgModule({
	declarations: [AppComponent, ToggleGroupComponent, ToggleButtonComponent],
	imports: [
		BrowserModule,
		AppRoutingModule,
		HomePageModule,
		BrowserAnimationsModule,
		HttpClientModule
	],
	providers: [
		IconRegistryService,
		{
			provide: APP_INITIALIZER,
			useFactory: (iconRegistryService: IconRegistryService) => () =>
				iconRegistryService.registerCustomIcons(),
			deps: [IconRegistryService],
			multi: true
		}
	],
	bootstrap: [AppComponent]
})
export class AppModule {}
