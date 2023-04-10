#if 0
//
// FelkelComponents.cpp
//
// Copyright (c) 2003-2011 Virtual Terrain Project
// Free for all uses, see license.txt for details.
//
// Straight skeleton algorithm and original implementation
// courtesy of Petr Felkel and Stepan Obdrzalek (petr.felkel@tiani.com)
// Re-implemented for the Virtual Terrain Project (vterrain.org)
// by Roger James (www.beardandsandals.co.uk)
//

#include "FelkelComponents.h"
#include "FelkelIntersection.h"

namespace StraightSkeleton
{
	//
	// Implementation of the CRidgeLine class.
	//

	CRidgeLine::CRidgeLine(const C3DPoint &p, const C3DPoint &q, const CNumber &Slope, const bool IsRidgeLine)
		: m_Origin(p)
	{
		m_Angle = atan2(q.m_z - p.m_z, q.m_x - p.m_x);
		m_Angle.NormalizeAngle();
		m_Slope = Slope;
		m_IsRidgeLine = IsRidgeLine;
	}

	CRidgeLine CRidgeLine::AngleAxis(const C3DPoint &b, const C3DPoint &a, const CNumber &sa, const C3DPoint &c, const CNumber &sc)
	{
		CNumber theta; // rotation from ba to bc
		CNumber theta1; // rotation from x-axis to ridgeline
		CNumber theta2; // slope of ridge line
		CRidgeLine ba(b, a, -1);
		CRidgeLine bc(b, c, -1);
		CNumber d1;
		CNumber d2;
		CNumber baAngle = ba.m_Angle;
		CNumber bcAngle = bc.m_Angle;
		// Clamp tans to first quadrant
		CNumber tsa = fabs(tan(sa));
		CNumber tsc = fabs(tan(sc));

		// Calculate the angle from ba to bc
		theta = bcAngle - baAngle;

		theta1 = atan2(sin(theta), (cos(theta) + tsa / tsc));

		theta2 = atan(sin(theta1) * tsa);

		if (theta2 < 0.0)
		{
			theta1 += CN_PI;
			theta2 = -theta2;
		}

		theta1 += ba.m_Angle;
		theta1.NormalizeAngle();

#ifdef FELKELDEBUG
		VTLOG("Angleaxis - aa %e sa %e ac %e sc %e theta %e theta1 %e theta2 %e\n",
			ba.m_Angle, sa, bc.m_Angle, sc, theta, theta1, theta2);
#endif
		// Create ridegline (origin, angle , slope, isRidgeline)
		return CRidgeLine(b, theta1, theta2, true);
	}

	C3DPoint CRidgeLine::Intersection(const CRidgeLine &a)
	{

		if (m_Origin == a.m_Origin)
			return m_Origin;
		if (PointOnRidgeLine(a.m_Origin) && a.PointOnRidgeLine(m_Origin))
		{
			if (m_IsRidgeLine && a.m_IsRidgeLine)
			{
#ifdef FELKELDEBUG
				VTLOG("COLLINEAR RIDGELINES ");
#endif
				if (m_Origin.m_y == a.m_Origin.m_y)
				{
#ifdef FELKELDEBUG
					VTLOG("(SAME HEIGHT)\n");
#endif
					if (m_Slope == a.m_Slope)
					{
						return C3DPoint(m_Origin.m_x + (a.m_Origin.m_x - m_Origin.m_x) / 2.0f,
							0,
							m_Origin.m_z + (a.m_Origin.m_z - m_Origin.m_z) / 2.0f);
					}
					if (m_Slope == 0.0)
					{
						return C3DPoint(a.m_Origin);
					}
					if (a.m_Slope == 0.0)
					{
						return C3DPoint(m_Origin);
					}

					return C3DPoint(m_Origin.m_x + (a.m_Origin.m_x - m_Origin.m_x) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))),
						0,
						m_Origin.m_z + (a.m_Origin.m_z - m_Origin.m_z) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))));
				}
				else
				{
					//				CNumber Offset;
					//				C3DPoint OffsetPoint;
#ifdef FELKELDEBUG
					VTLOG("(DIFFERENT HEIGHTS)\n");
#endif

					if (m_Origin.m_y < a.m_Origin.m_y)
					{
						CNumber OverallDistance = C3DPoint(a.m_Origin - m_Origin).LengthXZ();
						CNumber Height = a.m_Origin.m_y - m_Origin.m_y;
						CNumber TanA = tan(m_Slope);
						CNumber TanB = tan(a.m_Slope);
						CNumber Distance = (OverallDistance * TanA - Height) / (TanA + TanB);
						C3DPoint NewPoint;

						NewPoint = a.m_Origin + C3DPoint(cos(a.m_Angle) * Distance, 0.0, sin(a.m_Angle) * Distance);

						//					Offset = (a.m_Origin.m_y - m_Origin.m_y)/fabs(tan(m_Slope));
						//					OffsetPoint.m_x = Offset * cos(m_Angle) + m_Origin.m_x;
						//					OffsetPoint.m_z = Offset * sin(m_Angle) + m_Origin.m_z;
						//					C3DPoint OldPoint (m_Origin.m_x + Offset + (a.m_Origin.m_x - OffsetPoint.m_x) / (1 + fabs(tan(m_Slope))/fabs(tan(a.m_Slope))),
						//							0,
						//							m_Origin.m_z + Offset + (a.m_Origin.m_z - OffsetPoint.m_z) / (1 + fabs(tan(m_Slope))/fabs(tan(a.m_Slope))));

						return NewPoint;
					}
					else
					{
						CNumber OverallDistance = C3DPoint(a.m_Origin - m_Origin).LengthXZ();
						CNumber Height = m_Origin.m_y - a.m_Origin.m_y;
						CNumber TanA = tan(a.m_Slope);
						CNumber TanB = tan(m_Slope);
						CNumber Distance = (OverallDistance * TanA - Height) / (TanA + TanB);
						C3DPoint NewPoint;

						NewPoint = m_Origin + C3DPoint(cos(m_Angle) * Distance, 0.0, sin(m_Angle) * Distance);

						//					Offset = (m_Origin.m_y - a.m_Origin.m_y)/fabs(tan(a.m_Slope));
						//					OffsetPoint.m_x = Offset * cos(a.m_Angle) + a.m_Origin.m_x;
						//					OffsetPoint.m_z = Offset * sin(a.m_Angle) + a.m_Origin.m_z;
						//					C3DPoint OldPoint (m_Origin.m_x + (OffsetPoint.m_x - m_Origin.m_x) / (1 + tan(m_Slope)/tan(a.m_Slope)),
						//							0,
						//							m_Origin.m_z + (OffsetPoint.m_z - m_Origin.m_z) / (1 + fabs(tan(m_Slope))/fabs(tan(a.m_Slope))));

						return NewPoint;
					}
				}
			}
			else
			{
				CNumber OverallDistance = C3DPoint(a.m_Origin - m_Origin).LengthXZ();
				CNumber TanA = tan(m_Slope);
				CNumber TanB = tan(a.m_Slope);
				C3DPoint NewPoint;
				CNumber Distance = (OverallDistance * TanA) / (TanA + TanB);

				NewPoint = m_Origin + C3DPoint(cos(m_Angle) * Distance, 0.0, sin(m_Angle) * Distance);

				C3DPoint OldPoint((m_Origin.m_x + a.m_Origin.m_x) / 2, 0, (m_Origin.m_z + a.m_Origin.m_z) / 2);

				return NewPoint;
			}
		}
		if (PointOnRidgeLine(a.m_Origin))
			return a.m_Origin;
		if (a.PointOnRidgeLine(m_Origin))
			return m_Origin;
		if (Colinear(a))
			return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);

		CNumber sa = sin(m_Angle);
		CNumber sb = sin(a.m_Angle);
		CNumber ca = cos(m_Angle);
		CNumber cb = cos(a.m_Angle);
		CNumber x = sb*ca - sa*cb;

		if (x == 0.0)
			return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
		CNumber u = (cb*(m_Origin.m_z - a.m_Origin.m_z) - sb*(m_Origin.m_x - a.m_Origin.m_x)) / x;
		if (u != 0.0 && u < 0.0)
			return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
		if ((ca*(a.m_Origin.m_z - m_Origin.m_z) - sa*(a.m_Origin.m_x - m_Origin.m_x)) / x > 0)
			return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
		return C3DPoint(m_Origin.m_x + u*ca, 0, m_Origin.m_z + u*sa);
	};

	C3DPoint CRidgeLine::IntersectionAnywhere(const CRidgeLine& a) const
	{
		CRidgeLine OpaqueRidgeLine1, OpaqueRidgeLine2;

		if (m_Origin == a.m_Origin)
			return m_Origin;
		if (PointOnRidgeLine(a.m_Origin) && a.PointOnRidgeLine(m_Origin))
		{
			if (m_IsRidgeLine && a.m_IsRidgeLine)
			{
				assert(false);
				if (m_Origin.m_y == a.m_Origin.m_y)
				{
					return C3DPoint(m_Origin.m_x + (a.m_Origin.m_x - m_Origin.m_x) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))),
						0,
						m_Origin.m_z + (a.m_Origin.m_z - m_Origin.m_z) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))));
				}
				else
				{
					CNumber Offset;
					C3DPoint OffsetPoint;

					if (m_Origin.m_y < a.m_Origin.m_y)
					{
						Offset = (a.m_Origin.m_y - m_Origin.m_y) / fabs(tan(m_Slope));
						OffsetPoint.m_x = Offset * cos(m_Angle) + m_Origin.m_x;
						OffsetPoint.m_z = Offset * sin(m_Angle) + m_Origin.m_z;
						return C3DPoint(m_Origin.m_x + Offset + (a.m_Origin.m_x - OffsetPoint.m_x) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))),
							0,
							m_Origin.m_z + Offset + (a.m_Origin.m_z - OffsetPoint.m_z) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))));
					}
					else
					{
						Offset = (m_Origin.m_y - a.m_Origin.m_y) / fabs(tan(a.m_Slope));
						OffsetPoint.m_x = Offset * cos(a.m_Angle) + a.m_Origin.m_x;
						OffsetPoint.m_z = Offset * sin(a.m_Angle) + a.m_Origin.m_z;
						return C3DPoint(m_Origin.m_x + (OffsetPoint.m_x - m_Origin.m_x) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))),
							0,
							m_Origin.m_z + (OffsetPoint.m_z - m_Origin.m_z) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))));
					}
				}
			}
			else
				return C3DPoint((m_Origin.m_x + a.m_Origin.m_x) / 2, 0, (m_Origin.m_z + a.m_Origin.m_z) / 2);
		}
		if (PointOnRidgeLine(a.m_Origin))
			return a.m_Origin;
		if (a.PointOnRidgeLine(m_Origin))
			return m_Origin;
		if (Colinear(a))
			return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);

		OpaqueRidgeLine1 = Opaque();
		OpaqueRidgeLine2 = a.Opaque();

		if (OpaqueRidgeLine1.PointOnRidgeLine(a.m_Origin) && OpaqueRidgeLine2.PointOnRidgeLine(m_Origin))
		{
			if (m_IsRidgeLine && a.m_IsRidgeLine)
			{
				assert(false);
				if (m_Origin.m_y == a.m_Origin.m_y)
				{
					return C3DPoint(m_Origin.m_x + (a.m_Origin.m_x - m_Origin.m_x) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))),
						0,
						m_Origin.m_z + (a.m_Origin.m_z - m_Origin.m_z) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))));
				}
				else
				{
					CNumber Offset;
					C3DPoint OffsetPoint;

					if (m_Origin.m_y < a.m_Origin.m_y)
					{
						Offset = (a.m_Origin.m_y - m_Origin.m_y) / fabs(tan(m_Slope));
						OffsetPoint.m_x = Offset * cos(m_Angle) + m_Origin.m_x;
						OffsetPoint.m_z = Offset * sin(m_Angle) + m_Origin.m_z;
						return C3DPoint(m_Origin.m_x + Offset + (a.m_Origin.m_x - OffsetPoint.m_x) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))),
							0,
							m_Origin.m_z + Offset + (a.m_Origin.m_z - OffsetPoint.m_z) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))));
					}
					else
					{
						Offset = (m_Origin.m_y - a.m_Origin.m_y) / fabs(tan(a.m_Slope));
						OffsetPoint.m_x = Offset * cos(a.m_Angle) + a.m_Origin.m_x;
						OffsetPoint.m_z = Offset * sin(a.m_Angle) + a.m_Origin.m_z;
						return C3DPoint(m_Origin.m_x + (OffsetPoint.m_x - m_Origin.m_x) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))),
							0,
							m_Origin.m_z + (OffsetPoint.m_z - m_Origin.m_z) / (1 + fabs(tan(m_Slope)) / fabs(tan(a.m_Slope))));
					}
				}
			}
			else
				return C3DPoint((m_Origin.m_x + a.m_Origin.m_x) / 2, 0, (m_Origin.m_z + a.m_Origin.m_z) / 2);
		}

		CNumber sa = sin(m_Angle);
		CNumber sb = sin(a.m_Angle);
		CNumber ca = cos(m_Angle);
		CNumber cb = cos(a.m_Angle);
		CNumber x = sb*ca - sa*cb;

		if (x == 0.0)
			return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
		CNumber u = (cb*(m_Origin.m_z - a.m_Origin.m_z) - sb*(m_Origin.m_x - a.m_Origin.m_x)) / x;
		return C3DPoint(m_Origin.m_x + u*ca, 0, m_Origin.m_z + u*sa);
	};


	bool CRidgeLine::Colinear(const CRidgeLine &a) const
	{
		CNumber aa = m_Angle;
		CNumber ba = a.m_Angle;
		CNumber aa2 = m_Angle + CN_PI;

		aa.NormalizeAngle();
		ba.NormalizeAngle();
		aa2.NormalizeAngle();
		return (ba == aa || ba == aa2) ? true : false;
	}

	CNumber CRidgeLine::Dist(const C3DPoint &p) const
	{
		CNumber a = m_Angle - CRidgeLine(m_Origin, p, -1).m_Angle;
		CNumber d = sin(a) * m_Origin.DistXZ(p);
		return fabs(d);
	}


	//
	// implementation of the CNumber class.
	//

	CNumber::CNumber(double x)
	{
		m_n = x;
	}

	CNumber& CNumber::NormalizeAngle()
	{
		if (m_n >= CN_PI)
		{
			m_n = m_n - 2 * CN_PI;
			return NormalizeAngle();
		}
		if (m_n < -CN_PI)
		{
			m_n = m_n + 2 * CN_PI;
			return NormalizeAngle();
		}
		return *this;
	}

	CNumber CNumber::NormalizedAngle()
	{
		CNumber temp = *this;

		temp.NormalizeAngle();

		return temp;
	}

	// Implementation of the CVertex class.
	//

	//
	// Constructor
	//
	CVertex::CVertex(const C3DPoint &p, CVertex &left, CVertex &right)
		: m_point(p), m_done(false), m_higher(NULL), m_ID(-1), m_leftSkeletonLine(NULL), m_rightSkeletonLine(NULL),
		m_advancingSkeletonLine(NULL)
	{
		CNumber slope;

		m_leftLine = left.m_leftLine;
		m_rightLine = right.m_rightLine;
		m_leftVertex = &left;
		m_rightVertex = &right;

#if 1
		m_axis = CRidgeLine::AngleAxis(m_point,
			m_point + C3DPoint(cos(m_leftLine.m_Angle), m_point.m_y, sin(m_leftLine.m_Angle)), m_leftLine.m_Slope,
			m_point + C3DPoint(cos(m_rightLine.m_Angle), m_point.m_y, sin(m_rightLine.m_Angle)), m_rightLine.m_Slope);
#else
		// This lot assumes the slopes are equal !!!!!!!!!!
		// This needs fixing !!!!!!!!!!
		assert(m_leftLine.m_Slope == m_rightLine.m_Slope);

		// Create the associated ridgeline for this vertex
		CNumber height = m_point.m_y;

		C3DPoint i = m_leftLine.Intersection (m_rightLine);
		if (i.m_x == CN_INFINITY)
		{
#if VTDEBUG
			if (i.m_z != CN_INFINITY)
				VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif
			i = m_leftLine.IntersectionAnywhere (m_rightLine);
			if (i.m_x == CN_INFINITY)
			{
#if VTDEBUG
				if (i.m_z != CN_INFINITY)
					VTLOG("%s %d Assert failed i.m_z %e\n", __FILE__, __LINE__, i.m_z);
#endif
				if (m_leftLine.PointOnRidgeLine(m_rightLine.m_Origin) || m_leftLine.Opaque().PointOnRidgeLine(m_rightLine.m_Origin))
					// Lines are coincident
					m_axis = CRidgeLine(m_point, m_leftLine.m_Angle + CN_PI/2, m_leftLine.m_Slope);
				else
					// Lines are parallel
					m_axis = CRidgeLine(m_point, m_leftLine.m_Angle, 0);
			}
			else
			{
				// Reflex intersection
				CNumber Height2 = tan(m_leftLine.m_Slope) * m_leftLine.Dist(i);
				slope = - atan((Height2 - height) / m_point.DistXZ(i));
				m_axis = CRidgeLine(m_point, i, slope, true);
				m_axis.m_Angle = m_axis.m_Angle + CN_PI;
				m_axis.m_Angle.NormalizeAngle();
			}
		}
		else
		{
			CNumber Height2 = tan(m_leftLine.m_Slope) * m_leftLine.Dist(i);
			slope = atan((Height2 - height) / m_point.DistXZ(i));
			m_axis = CRidgeLine(m_point, i, slope, true);
		}
#endif
	}

	C3DPoint CVertex::CoordinatesOfAnyIntersectionOfTypeB(const CVertex &left, const CVertex &right)
	{

		CRidgeLine Reverse;
		C3DPoint p1 = m_rightLine.IntersectionAnywhere(right.m_leftLine);
		C3DPoint p2 = m_leftLine.IntersectionAnywhere(left.m_rightLine);
		C3DPoint poi(CN_INFINITY, CN_INFINITY, CN_INFINITY);

		if (p1 != C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY) && p2 != C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY))
		{
			if (m_rightLine.PointOnRidgeLine(p1)) return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
			if (m_leftLine.PointOnRidgeLine(p2))  return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
		}

		poi = left.m_rightLine.IntersectionAnywhere(m_axis);
#if VTDEBUG
		if (poi.m_y != 0.0)
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (left.m_rightLine.m_IsRidgeLine)
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif

		Reverse = CRidgeLine(poi, m_point, 0, true);
		Reverse.m_Slope = atan(fabs(tan(left.m_rightLine.m_Slope)) * sin(left.m_rightLine.m_Angle - Reverse.m_Angle));

		poi = m_axis.Intersection(Reverse);

		// Calculate heights
		poi.m_y = m_leftLine.Dist(poi) * fabs(tan(m_leftLine.m_Slope));

#if VTDEBUG
		{
			if (poi.m_y < 10000.0)
			{
				CNumber db0 = (poi - m_axis.m_Origin).LengthXZ() * fabs(tan(m_axis.m_Slope));
				CNumber db1 = m_rightLine.Dist(poi) * fabs(tan(m_rightLine.m_Slope));
				CNumber db2 = left.m_rightLine.Dist(poi) * fabs(tan(left.m_rightLine.m_Slope));
				CNumber db3 = left.m_nextVertex->m_leftLine.Dist(poi) * fabs(tan(left.m_nextVertex->m_leftLine.m_Slope));
				if (!SIMILAR(poi.m_y, db0))
					VTLOG("%s %d Assert failed poi.m_y %e db0 %e\n", __FILE__, __LINE__, double(poi.m_y), double(db0));
				if (!SIMILAR(poi.m_y, db1))
					VTLOG("%s %d Assert failed poi.m_y %e db1 %e\n", __FILE__, __LINE__, double(poi.m_y), double(db1));
				if (!SIMILAR(poi.m_y, db2))
					VTLOG("%s %d Assert failed poi.m_y %e db2 %e\n", __FILE__, __LINE__, double(poi.m_y), double(db2));
				if (!SIMILAR(poi.m_y, db3))
					VTLOG("%s %d Assert failed poi.m_y %e db3 %e\n", __FILE__, __LINE__, double(poi.m_y), double(db3));
			}
		}
#endif
		return poi;
	}

	C3DPoint CVertex::IntersectionOfTypeB(const CVertex &left, const CVertex &right)
	{
#if VTDEBUG
		if (m_prevVertex != NULL && !m_leftLine.FacingTowards(m_prevVertex->m_rightLine))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (m_nextVertex != NULL && !m_rightLine.FacingTowards(m_nextVertex->m_leftLine))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (left.m_prevVertex != NULL && !left.m_leftLine.FacingTowards(left.m_prevVertex->m_rightLine))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (left.m_nextVertex != NULL && !left.m_rightLine.FacingTowards(left.m_nextVertex->m_leftLine))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (right.m_prevVertex != NULL && !right.m_leftLine.FacingTowards(right.m_prevVertex->m_rightLine))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (right.m_nextVertex != NULL && !right.m_rightLine.FacingTowards(right.m_nextVertex->m_leftLine))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif

		C3DPoint pl(m_axis.Intersection(left.m_rightLine));
		C3DPoint pr(m_axis.Intersection(right.m_leftLine));
		if (pl == C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY) && pr == C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY))
			return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);

		C3DPoint p;
		if (pl != C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY)) p = pl;
		if (pr != C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY)) p = pr;
#if VTDEBUG
		if (p == C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
		if (!(pl == C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY) || pr == C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY) || pl == pr))
			VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif

		C3DPoint poi = CoordinatesOfAnyIntersectionOfTypeB(left, right);
		CNumber al = left.m_axis.m_Angle - left.m_rightLine.m_Angle;
		CNumber ar = right.m_axis.m_Angle - right.m_leftLine.m_Angle;

		CNumber alp = CRidgeLine(left.m_point, poi).m_Angle - left.m_rightLine.m_Angle;
		CNumber arp = CRidgeLine(right.m_point, poi).m_Angle - right.m_leftLine.m_Angle;

		al.NormalizeAngle(); ar.NormalizeAngle(); alp.NormalizeAngle(); arp.NormalizeAngle();
#if VTDEBUG
		if (!(al <= 0.0))
			VTLOG("%s %d Assert failed al %e\n", __FILE__, __LINE__, double(al));
		if (!(ar >= 0.0 || ar == -CN_PI))
			VTLOG("%s %d Assert failed ar %e\n", __FILE__, __LINE__, double(ar));
#endif

		if ((alp > 0.0 || alp < al) && !ANGLE_SIMILAR(alp, CNumber(0)) && !ANGLE_SIMILAR(alp, al))
			return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
		if ((arp < 0.0 || arp > ar) && !ANGLE_SIMILAR(arp, CNumber(0)) && !ANGLE_SIMILAR(arp, ar))
			return C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
		return poi;
	}

	CNumber CVertex::NearestIntersection(CVertexList &vl, CVertex **left, CVertex **right, C3DPoint &p)
	{
		CNumber minDist = CN_INFINITY;
		CVertexList::iterator minI = vl.end();
		CVertexList::iterator i;
		for (i = vl.begin(); i != vl.end(); i++)
		{
#ifdef FELKELDEBUG
			CVertex TempVertex = *i;
#endif
			if ((*i).m_done) continue;
			if ((*i).m_nextVertex == NULL || (*i).m_prevVertex == NULL) continue;
			if (&*i == this || (*i).m_nextVertex == this) continue;
#if VTDEBUG
			if (!((*i).m_rightVertex != NULL))
				VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
			if (!((*i).m_leftVertex != NULL))
				VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif
			C3DPoint poi = IntersectionOfTypeB((*i), *(*i).m_nextVertex);
			if (poi == C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY)) continue;
			CNumber d = poi.DistXZ(m_point);
			if (d < minDist) { minDist = d; minI = i; }
		}
		if (minDist == CN_INFINITY)
		{
			p = C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY);
			return CN_INFINITY;
		}

		i = minI;
		C3DPoint poi = CoordinatesOfAnyIntersectionOfTypeB((*i), *(*i).m_nextVertex);

		p = poi;
		*left = (CVertex *)&*i;
		*right = (*i).m_nextVertex;

		return poi.DistXZ(m_point);
	}

	bool CVertex::InvalidIntersection(CVertexList &vl, const CIntersection &is)
	{
		for (CVertexList::iterator i = vl.begin(); i != vl.end(); i++)
		{
			if ((*i).m_done)
				continue;
			if ((*i).m_nextVertex == NULL || (*i).m_prevVertex == NULL)
				continue;

			C3DPoint poi = m_axis.Intersection((*i).m_axis);
			if (poi == C3DPoint(CN_INFINITY, CN_INFINITY, CN_INFINITY))
				continue;
			if (&*i == is.m_leftVertex || &*i == is.m_rightVertex)
				continue;
			if (m_axis.PointOnRidgeLine((*i).m_axis.m_Origin) && (*i).m_axis.PointOnRidgeLine(m_axis.m_Origin))
				continue;

			// Calculate height of intersection
			CNumber dv = m_leftLine.Dist(poi) * fabs(tan(m_leftLine.m_Slope));
			CNumber dvx = m_rightLine.Dist(poi) * fabs(tan(m_rightLine.m_Slope));
#ifdef FELKELDEBUG
			if (!(dv == dvx))
				VTLOG("%s %d Assert failed dv %e dvx %e\n", __FILE__, __LINE__, dv, dvx);
#endif
			if (dv > is.m_height)
				continue;

			CNumber di = (*i).m_leftLine.Dist(poi) * fabs(tan((*i).m_leftLine.m_Slope));
			CNumber dix = (*i).m_rightLine.Dist(poi) * fabs(tan((*i).m_rightLine.m_Slope));
#if VTDEBUG
			if (!(di == dix))
				VTLOG("%s %d Assert failed\n", __FILE__, __LINE__);
#endif
			if (di > dv + MIN_DIFF)
				continue;

			return true;
		}
		return false;
	}

	bool CVertex::VertexInCurrentContour(CVertex& Vertex)
	{
		CVertex* ActiveVertex = m_nextVertex;

		if (NULL == ActiveVertex)
			return false;

		while (ActiveVertex->m_ID != m_ID)
		{
			if (Vertex.m_ID == ActiveVertex->m_ID)
				return true;
			//		ActiveVertex = ActiveVertex->m_rightVertex->m_ID == ActiveVertex->m_ID ? ActiveVertex->m_nextVertex : ActiveVertex->m_rightVertex;
			ActiveVertex = ActiveVertex->m_nextVertex;
		}
		return false;
	}
}

#endif
