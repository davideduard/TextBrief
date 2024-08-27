import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { HomePageRoutingModule } from './home-page-routing.module';

import * as fromComponents from './components';
import * as fromContainers from './containers';
import { HomePageContainer } from './containers';
import { FlatButtonComponent } from '../../shared/components';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatIconModule } from '@angular/material/icon';
import { FormsModule } from '@angular/forms';

@NgModule({
	declarations: [...fromComponents.components, ...fromContainers.containers],
	exports: [HomePageContainer],
	imports: [
		CommonModule,
		HomePageRoutingModule,
		FlatButtonComponent,
		MatButtonToggleModule,
		MatIconModule,
		FormsModule
	]
})
export class HomePageModule {}
