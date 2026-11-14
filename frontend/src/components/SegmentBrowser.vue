<script setup lang="ts">
import {ref,computed,watch} from 'vue';
import {researchApi} from '../api/research';
import {fmt} from '../utils/format';
import Table from './common/Table.vue';
const p=defineProps<{runId:number;trace:any;states:string[]}>();
const meta=ref<any>(null);
const block=ref<any>(null);
const index=ref(0);
const loading=ref(false);
const failed=ref('');
const segmented=computed(()=>Array.isArray(p.trace.blocks)&&p.trace.blocks.length>0);
async function loadMeta(){
 meta.value=null;block.value=null;index.value=0;failed.value='';
 try{meta.value=await researchApi.segments(p.runId,p.trace.id);await loadBlock(0);}
 catch(e:any){failed.value=e.message;}
}
async function loadBlock(i:number){
 if(!meta.value||i<0||i>=meta.value.block_count)return;
 loading.value=true;failed.value='';
 try{block.value=await researchApi.segment(p.runId,p.trace.id,i);index.value=i;}
 catch(e:any){failed.value=e.message;}
 finally{loading.value=false;}
}
watch(()=>p.trace.id,loadMeta,{immediate:true});
function bandPath(path:number[],start:number){return path.slice(start,start+(block.value?.size||0));}
</script>
<template>
 <section class="segment-browser" v-if="segmented">
  <div class="section-head">
   <h2>长轨迹分段查看（整序列一次推断，区段仅为查看单位）</h2>
   <span>{{trace.length}} 码 / {{trace.block_count}} 区段，每段 {{trace.block_size}} 码</span>
  </div>
  <p v-if="failed" class="copper">区段加载失败：{{failed}}</p>
  <template v-if="meta">
   <div class="segment-metrics">
    <span>序列logL <strong>{{fmt(meta.log_likelihood,8)}}</strong></span>
    <span>联合MAP logP <strong>{{fmt(meta.viterbi_log_joint,8)}}</strong></span>
    <span>{{meta.linear_likelihood_underflows?'线性概率下溢，logL仍有限':'线性概率 '+meta.likelihood_if_representable.toExponential(6)}}</span>
    <span :class="{copper:!meta.marginal_path_is_legal}">{{meta.marginal_path_is_legal?'边际拼接路径合法（仍非联合定义）':'边际拼接违反零概率约束'}}</span>
   </div>
   <div class="segment-pager">
    <button type="button" :disabled="index===0||loading" @click="loadBlock(index-1)">← 上一区段</button>
    <label>区段
     <select :value="index" @change="loadBlock(Number(($event.target as HTMLSelectElement).value))">
      <option v-for="b in meta.blocks" :key="b.index" :value="b.index">#{{b.index+1}} · 步 {{b.start+1}}–{{b.end}}</option>
     </select>
    </label>
    <button type="button" :disabled="index>=meta.block_count-1||loading" @click="loadBlock(index+1)">下一区段 →</button>
    <small>{{loading?'加载中…':`当前区段 #${index+1}，步 ${block?block.start+1:0}–${block?block.end:0}`}}</small>
   </div>
   <div class="band-scroll" v-if="block">
    <div class="band-table" :style="{minWidth:Math.max(500,block.size*24+125)+'px'}">
     <div class="band-row">
      <small>步号 / 观测</small>
      <span v-for="(token,u) in block.tokens" :key="u" class="token-cell"><em>{{block.start+u+1}}</em>{{token}}</span>
     </div>
     <div class="band-row">
      <small>Viterbi 联合MAP</small>
      <span v-for="(s,u) in bandPath(meta.viterbi_path,block.start)" :key="u" :class="['state-cell','state-'+s]" :title="states[s]">{{s}}</span>
     </div>
     <div class="band-row">
      <small>单点最高后验</small>
      <span v-for="(s,u) in bandPath(meta.marginal_path,block.start)" :key="u" :class="['state-cell','state-'+s]" :title="states[s]">{{s}}</span>
     </div>
    </div>
   </div>
   <div class="posterior-scroll" v-if="block">
    <div class="posterior-table" :style="{minWidth:Math.max(500,block.size*24+125)+'px'}">
     <div class="posterior-row" v-for="(state,s) in states" :key="state">
      <strong>{{s}} / {{state}}</strong>
      <span v-for="(row,u) in block.posterior" :key="u" class="posterior-cell"
       :style="{backgroundColor:'oklch(78% .08 88 / '+row[s]+')'}"
       :title="'t'+(block.start+u+1)+': '+fmt(row[s],8)"></span>
     </div>
    </div>
   </div>
   <details class="provenance" v-if="block">
    <summary>本区段完整 log 数组（null 表示精确零概率 -∞）</summary>
    <Table :headers="['步','log forward','log backward','Viterbi父指针']">
     <tr v-for="(row,u) in block.log_forward" :key="u">
      <td>{{block.start+u+1}}</td>
      <td>{{JSON.stringify(row)}}</td>
      <td>{{JSON.stringify(block.log_backward[u])}}</td>
      <td>{{JSON.stringify(block.viterbi_parent[u])}}</td>
     </tr>
    </Table>
    <p>{{meta.tie_rule}}</p>
    <p>{{meta.null_log_meaning}}</p>
   </details>
  </template>
 </section>
</template>
