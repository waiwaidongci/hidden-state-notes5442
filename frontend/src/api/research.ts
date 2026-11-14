import {request,post,patch} from './client';
export const researchApi={
 runs:()=>request('/runs'),
 notes:()=>request('/notes'),
 createRun:(data:any)=>post('/runs',data),
 createNote:(data:any)=>post('/notes',data),
 approve:(id:number,data:any)=>patch('/notes/'+id+'/approval',data),
 segments:(runId:number,traceId:number)=>request(`/runs/${runId}/traces/${traceId}/segments`),
 segment:(runId:number,traceId:number,index:number)=>request(`/runs/${runId}/traces/${traceId}/segments/${index}`),
};
