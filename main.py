from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="웹툰 분석 API",
    description="웹툰 데이터 분석 및 추천 시스템 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델
class WebtoonData(BaseModel):
    rank: int
    title: str
    tags: List[str]
    interest_count: int
    rating: float
    gender: str
    ages: str

class RecommendationRequest(BaseModel):
    title: str
    limit: Optional[int] = 5

class AnalysisResponse(BaseModel):
    success: bool
    data: Optional[Dict] = None
    message: Optional[str] = None

# 샘플 데이터 (나중에 실제 DB로 교체)
SAMPLE_WEBTOONS = [
    {"rank": 1, "title": "화산귀환", "tags": ["회귀", "무협/사극", "액션", "명작"], "interest_count": 1534623, "rating": 9.88, "gender": "남성", "ages": "20대"},
    {"rank": 2, "title": "신의 탑", "tags": ["이능력", "액션", "판타지", "성장물"], "interest_count": 1910544, "rating": 9.84, "gender": "남성", "ages": "20대"},
    {"rank": 3, "title": "외모최강주의", "tags": ["드라마", "학원액션", "소년물", "격투기"], "interest_count": 824399, "rating": 9.40, "gender": "남성", "ages": "10대"},
    {"rank": 4, "title": "마른 가지에 바람처럼", "tags": ["로맨스", "순정남", "서양", "왕족/귀족"], "interest_count": 458809, "rating": 9.97, "gender": "여성", "ages": "10대"},
    {"rank": 5, "title": "엄마를 만나러 가는 길", "tags": ["판타지", "모험", "감성적인", "러블리"], "interest_count": 259146, "rating": 9.98, "gender": "여성", "ages": "10대"},
    {"rank": 6, "title": "재혼 황후", "tags": ["로맨스", "왕족/귀족", "서양", "복수"], "interest_count": 892456, "rating": 9.75, "gender": "여성", "ages": "20대"},
    {"rank": 7, "title": "나 혼자만 레벨업", "tags": ["액션", "게임", "이능력", "성장물"], "interest_count": 2156789, "rating": 9.91, "gender": "남성", "ages": "20대"},
    {"rank": 8, "title": "여신강림", "tags": ["로맨스", "학원", "일상", "개그"], "interest_count": 1345678, "rating": 9.62, "gender": "여성", "ages": "10대"},
    {"rank": 9, "title": "이태원 클라쓰", "tags": ["드라마", "현실", "성장물", "사이다"], "interest_count": 987654, "rating": 9.55, "gender": "남성", "ages": "30대"},
    {"rank": 10, "title": "유미의 세포들", "tags": ["로맨스", "일상", "감성드라마", "공감"], "interest_count": 756432, "rating": 9.33, "gender": "여성", "ages": "30대"},
    {"rank": 11, "title": "전지적 독자 시점", "tags": ["회귀", "판타지", "액션", "성장물"], "interest_count": 1823456, "rating": 9.92, "gender": "남성", "ages": "20대"},
    {"rank": 12, "title": "악역의 엔딩은 죽음뿐", "tags": ["로맨스", "회귀", "판타지", "왕족/귀족"], "interest_count": 734521, "rating": 9.78, "gender": "여성", "ages": "20대"},
    {"rank": 13, "title": "나의 수학선생", "tags": ["로맨스", "학원", "드라마", "일상"], "interest_count": 654321, "rating": 9.45, "gender": "여성", "ages": "20대"},
    {"rank": 14, "title": "대학원 탈출일지", "tags": ["일상", "개그", "현실", "공감"], "interest_count": 543210, "rating": 9.23, "gender": "남성", "ages": "20대"},
    {"rank": 15, "title": "기기괴괴", "tags": ["호러", "스릴러", "단편", "미스터리"], "interest_count": 432109, "rating": 9.34, "gender": "남성", "ages": "20대"},
]

def load_webtoon_data():
    """웹툰 데이터 로드 (나중에 DB 연결로 교체)"""
    return SAMPLE_WEBTOONS

def calculate_tag_frequency(webtoons_data):
    """태그 빈도 계산"""
    tag_count = {}
    for webtoon in webtoons_data:
        for tag in webtoon['tags']:
            tag_count[tag] = tag_count.get(tag, 0) + 1
    
    return sorted(tag_count.items(), key=lambda x: x[1], reverse=True)

def calculate_tag_cooccurrence(webtoons_data):
    """태그 동시 출현 계산"""
    cooccurrence = {}
    
    for webtoon in webtoons_data:
        tags = webtoon['tags']
        for i, tag1 in enumerate(tags):
            for tag2 in tags[i+1:]:
                pair = tuple(sorted([tag1, tag2]))
                cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
    
    return cooccurrence

def get_recommendations_by_tags(target_title, webtoons_data, limit=5):
    """태그 유사도 기반 추천"""
    target_webtoon = next((w for w in webtoons_data if w['title'] == target_title), None)
    if not target_webtoon:
        return []
    
    target_tags = set(target_webtoon['tags'])
    recommendations = []
    
    for webtoon in webtoons_data:
        if webtoon['title'] == target_title:
            continue
            
        webtoon_tags = set(webtoon['tags'])
        
        # Jaccard 유사도 계산
        intersection = len(target_tags & webtoon_tags)
        union = len(target_tags | webtoon_tags)
        similarity = intersection / union if union > 0 else 0
        
        if similarity > 0:
            recommendations.append({
                **webtoon,
                'similarity': similarity,
                'common_tags': list(target_tags & webtoon_tags)
            })
    
    # 유사도 순으로 정렬
    recommendations.sort(key=lambda x: x['similarity'], reverse=True)
    return recommendations[:limit]

def generate_heatmap_data(webtoons_data):
    """히트맵 데이터 생성"""
    genres = ['로맨스', '액션', '판타지', '드라마', '무협/사극', '일상']
    demographics = ['남성-10대', '남성-20대', '남성-30대', '여성-10대', '여성-20대', '여성-30대']
    
    heatmap_data = []
    
    for demo_idx, demo in enumerate(demographics):
        gender, age = demo.split('-')
        for genre_idx, genre in enumerate(genres):
            count = sum(1 for w in webtoons_data 
                       if w['gender'] == gender and w['ages'] == age and genre in w['tags'])
            
            heatmap_data.append({
                'x': genre_idx,
                'y': demo_idx,
                'value': count,
                'genre': genre,
                'demographic': demo,
                'count': count
            })
    
    return heatmap_data

# API 엔드포인트들

@app.get("/")
async def read_root():
    return {
        "message": "웹툰 분석 API 서버가 정상 작동 중입니다",
        "version": "1.0.0",
        "endpoints": {
            "webtoons": "/api/webtoons",
            "tag_analysis": "/api/analysis/tags",
            "heatmap": "/api/analysis/heatmap",
            "recommendations": "/api/recommendations",
            "statistics": "/api/stats"
        }
    }

@app.get("/api/webtoons")
async def get_webtoons():
    """모든 웹툰 데이터 반환"""
    try:
        data = load_webtoon_data()
        return {"success": True, "data": data, "count": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 로딩 실패: {str(e)}")

@app.get("/api/analysis/tags")
async def get_tag_analysis():
    """태그 분석 데이터 반환"""
    try:
        webtoons_data = load_webtoon_data()
        
        # 태그 빈도
        tag_frequency = calculate_tag_frequency(webtoons_data)
        
        # 태그 동시 출현
        tag_cooccurrence = calculate_tag_cooccurrence(webtoons_data)
        
        # 네트워크 노드 생성
        network_nodes = [{'id': tag, 'count': count, 'size': count * 5} 
                        for tag, count in tag_frequency[:20]]
        
        # 네트워크 링크 생성
        network_links = [{'source': pair[0], 'target': pair[1], 'value': count}
                        for pair, count in tag_cooccurrence.items() if count >= 2]
        
        return {
            "success": True,
            "data": {
                "tag_frequency": tag_frequency[:15],
                "network_nodes": network_nodes,
                "network_links": network_links
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"태그 분석 실패: {str(e)}")

@app.get("/api/analysis/heatmap")
async def get_heatmap_analysis():
    """히트맵 분석 데이터 반환"""
    try:
        webtoons_data = load_webtoon_data()
        heatmap_data = generate_heatmap_data(webtoons_data)
        
        return {"success": True, "data": heatmap_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"히트맵 분석 실패: {str(e)}")

@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """추천 웹툰 반환"""
    try:
        webtoons_data = load_webtoon_data()
        recommendations = get_recommendations_by_tags(
            request.title, 
            webtoons_data, 
            request.limit
        )
        
        return {
            "success": True, 
            "data": recommendations,
            "count": len(recommendations),
            "requested_title": request.title
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추천 생성 실패: {str(e)}")

@app.get("/api/stats")
async def get_statistics():
    """전체 통계 반환"""
    try:
        webtoons_data = load_webtoon_data()
        
        total_webtoons = len(webtoons_data)
        avg_rating = np.mean([w['rating'] for w in webtoons_data])
        avg_interest = np.mean([w['interest_count'] for w in webtoons_data])
        
        # 고유 태그 수
        all_tags = set()
        for w in webtoons_data:
            all_tags.update(w['tags'])
        unique_tags = len(all_tags)
        
        # 성별/연령별 분포
        gender_dist = {}
        age_dist = {}
        for w in webtoons_data:
            gender_dist[w['gender']] = gender_dist.get(w['gender'], 0) + 1
            age_dist[w['ages']] = age_dist.get(w['ages'], 0) + 1
        
        return {
            "success": True,
            "data": {
                "total_webtoons": total_webtoons,
                "avg_rating": round(avg_rating, 2),
                "avg_interest": int(avg_interest),
                "unique_tags": unique_tags,
                "gender_distribution": gender_dist,
                "age_distribution": age_dist,
                "last_updated": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 계산 실패: {str(e)}")

@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    # 개발용 설정
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,  # 코드 변경시 자동 재시작
        log_level="info"
    )