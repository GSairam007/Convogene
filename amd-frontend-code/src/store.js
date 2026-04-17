// store.js
import { createStore, combineReducers } from 'redux';
import dataReducer from './reducers/dataReducer';

const rootReducer = combineReducers({
    data: dataReducer,
    // Add other reducers here if needed
});

const store = createStore(rootReducer);

export default store;
