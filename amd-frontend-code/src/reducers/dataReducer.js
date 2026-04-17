// dataActions.js
export const setData = (data) => {
    return {
        type: 'SET_DATA',
        recordedVideos: data,
    };
};

// dataReducer.js
const initialState = {
    recordedVideos: [],
};

const dataReducer = (state = initialState, action) => {
    switch (action.type) {
        case 'SET_DATA':
            // The place where redux response returns
            return { ...state, recordedVideos: action.recordedVideos };
        default:
            return state;
    }
};

export default dataReducer;
