/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http: *www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
**/

'use strict';

describe('OCW Controllers', function() {

	beforeEach(module('ocw.controllers'));
	beforeEach(module('ocw.services'));

	describe('DatasetDisplayCtrl', function() {
		it('should initialize the removeDataset function', function() {
			inject(function($rootScope, $controller) {
				var scope = $rootScope.$new();
				var ctrl = $controller("DatasetDisplayCtrl", {$scope: scope});

				scope.datasets.push(1);
				scope.datasets.push(2);

				expect(scope.datasets[0]).toBe(1);

				scope.removeDataset(0);

				expect(scope.datasets[0]).toBe(2);
			});
		});

		it('should initialize the setRegridBase function', function() {
			inject(function($rootScope, $controller) {
				var scope = $rootScope.$new();
				var ctrl = $controller("DatasetDisplayCtrl", {$scope: scope});

				scope.datasets.push({regrid: false});
				scope.datasets.push({regrid: true});
				scope.datasets.push({regrid: true});
				scope.datasets.push({regrid: false});

				expect(scope.datasets[1].regrid).toBe(true);
				expect(scope.datasets[2].regrid).toBe(true);
				
				// setRegridBase should set all indices to 'false' if if it's not the passed index
			    scope.setRegridBase(2);

				expect(scope.datasets[1].regrid).toBe(false);
				expect(scope.datasets[2].regrid).toBe(true);
			});
		});
	});
});
