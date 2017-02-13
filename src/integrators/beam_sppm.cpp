
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


// integrators/beam_sppm.cpp*
#include "integrators/beam_sppm.h"
#include "bssrdf.h"
#include "parallel.h"
#include "scene.h"
#include "imageio.h"
#include "spectrum.h"
#include "rng.h"
#include "paramset.h"
#include "progressreporter.h"
#include "interaction.h"
#include "sampling.h"
#include "samplers/halton.h"
#include "stats.h"

STAT_TIMER("Time/Beam_SPPM camera pass", hitPointTimer);
STAT_TIMER("Time/Beam_SPPM visible point grid construction", 
            gridConstructionTimer);
STAT_TIMER("Time/Beam_SPPM photon pass", photonTimer);
STAT_TIMER("Time/Beam_SPPM statistics update", statsUpdateTimer);
STAT_RATIO("Stochastic Progressive Photon Mapping/Visible points checked per photon intersection", 
            visiblePointsChecked, totalPhotonSurfaceInteractions);
STAT_COUNTER("Subsurface Scattering Stochastic Progressive Photon Mapping/Photon paths followed",
             photonPaths);
STAT_INT_DISTRIBUTION(
    "Subsurface Scattering Stochastic Progressive Photon Mapping/Grid cells per visible point",
    gridCellsPerVisiblePoint);
STAT_MEMORY_COUNTER("Memory/Beam_SPPM Pixels", pixelMemoryBytes);
STAT_FLOAT_DISTRIBUTION("Memory/Beam_SPPM BSDF and Grid Memory", memoryArenaMB);

// Beam_SPPM Local Definitions
struct Beam_SPPMPixel {
    // Beam_SPPMPixel Public Methods
    Beam_SPPMPixel() : M(0) {}

    // Beam_SPPMPixel Public Data
    Float radius = 0;
    Spectrum Ld;
    struct VisiblePoint {
        // VisiblePoint Public Methods
        VisiblePoint() {}
        VisiblePoint(const Point3f &p, const Vector3f &wo, const Spectrum &beta,
                     const BSDF *bsdf, const BSSRDF *bssrdf=nullptr)
            : p(p), wo(wo), bsdf(bsdf), bssrdf(bssrdf), beta(beta) {
                isSubsurface = (bssrdf == nullptr);
            }
        Point3f p;
        Vector3f wo;
        const BSDF *bsdf = nullptr;
        const BSSRDF *bssrdf = nullptr;
        Spectrum beta;
        bool isSubsurface = false;
    } vp;
    AtomicFloat Phi[Spectrum::nSamples];
    std::atomic<int> M;
    Float N = 0;
    Spectrum tau;
};

struct Beam_SPPMPixelListNode {
    Beam_SPPMPixel *pixel;
    Beam_SPPMPixelListNode *next;
};

static bool ToGrid(const Point3f &p, const Bounds3f &bounds,
                   const int gridRes[3], Point3i *pi) {
    bool inBounds = true;
    Vector3f pg = bounds.Offset(p);
    for (int i = 0; i < 3; ++i) {
        (*pi)[i] = (int)(gridRes[i] * pg[i]);
        inBounds &= ((*pi)[i] >= 0 && (*pi)[i] < gridRes[i]);
        (*pi)[i] = Clamp((*pi)[i], 0, gridRes[i] - 1);
    }
    return inBounds;
}

inline unsigned int hash(const Point3i &p, int hashSize) {
    return (unsigned int)((p.x * 73856093) ^ (p.y * 19349663) ^
                          (p.z * 83492791)) %
           hashSize;
}

Vector3f rotate(Vector3f w, Float theta, Float phi){
    Normal3f n;
    if (w.z > 0) {
        n = Normal3f(1., 1., (-w.x-w.y)/w.z);
    } else if (w.y > 0) {
        n = Normal3f(1., (-w.x-w.z)/w.y, 1.);
    } else {
        n = Normal3f((-w.y-w.z)/w.x, 1., 1.);
    }
    Transform rot = Rotate(phi, w);
    n = rot(n);
    return std::cos(theta) * w + Vector3f(std::sin(theta) * n); 
}

bool Beam_SPPMIntegrator::traceSubsurfaceScattering(SurfaceInteraction isect, 
                                                    const Scene &scene, 
                                                    Vector3f wo, 
                                                    Point3f o,
                                                    Vector3f *wi,
                                                    int &haltonDim, 
                                                    uint64_t haltonIndex, 
                                                    int maxDepth){
    // Refract wo as is enters the surface
    Vector3f currDir;
    BSSRDF *bssrdf = isect.bssrdf;
    printf("\nEntered with direction: %.3f, %.3f, %.3f\n", 
            -isect.wo.x, -isect.wo.y, -isect.wo.z);
    printf("Surface normal: %.3f, %.3f, %.3f\n", 
            isect.n.x, isect.n.y, isect.n.z);
    printf("Surface eta: %.3f\n", bssrdf->Eta());
    // Inverse intersection direction so it points towards the medium!
    bool refracted = Refract(-isect.wo, isect.n, bssrdf->Eta(), &currDir);
    if(!refracted){
        printf("Imposible to refract!\n");
        return true;
    }
    Vector3f normCurrDir = Normalize(currDir);
    printf("Refracted and normalized to: %.3f, %.3f, %.3f\n", 
            normCurrDir.x, normCurrDir.y, normCurrDir.z);
    // Advance a litte in refracted direction
    Point3f currPos = isect.p + Normalize(currDir) * 0.0001;
    printf("Current position: %.4f, %.4f, %.4f\n", 
            isect.p.x, isect.p.y, isect.p.z);
    printf("Current dislocated position: %.4f, %.4f, %.4f\n", 
            currPos.x, currPos.y, currPos.z);


    // TODO: Sample which channel to use
    int ch = 0;
    for(int depth = 0; depth < maxDepth; ++depth) {
        printf("depth: %d\n", depth);
        // Importance sample dis tance until next interaction
        Float xi = RadicalInverse(haltonDim++, haltonIndex);
        Float d = - std::log(xi) / bssrdf->Sigma_tCh(ch);
        printf("Sampled distance: %.3f\n", d);
        
        // Check if photon intersects medium surface
        SurfaceInteraction tisect;
        Ray ray = Ray(currPos, currDir);
        if (!scene.Intersect(ray, &tisect)){
            printf("Intersection with medium not found!\n");
            return true;
        }
        
        Float distTestisect = DistanceSquared(currPos, tisect.p);
        printf("Distance to surface intersection: %.3f\n", distTestisect);
        if (distTestisect <= d*d){
            printf("Sampled distance went out of material!\n");
            //Refract direction and   exit 
            bool refracted = Refract(currDir, -isect.n, 1 / bssrdf->Eta(), wi);
            if (!refracted) 
                return true;
            return false;
        }
        // Russian Roulette interaction type
        Float alpha = bssrdf->Sigma_sCh(ch) / bssrdf->Sigma_tCh(ch);
        xi = RadicalInverse(haltonDim, haltonIndex);
        if (xi > alpha){
            printf("Photon absorbed: %.3f < %.3f\n", xi, alpha);
            return true; // Photon is absorbed
        }
        printf("Photon scattered:%.3f > %.3f\n", xi, alpha);
        
        // Photon is scattered, importance sample direction
        Float g = bssrdf->G();
        xi = RadicalInverse(haltonDim + 1, haltonIndex);
        Float theta = 2 * Pi * xi;
        if (g != 0.){
            Float gg = g*g;
            Float t = (1 - gg) / (1 - g + 2 * g * xi);
            theta = std::acos(1 / (2 * g) * (1 + gg - t*t));
        }
        
        xi = RadicalInverse(haltonDim + 2, haltonIndex);
        Float phi = xi * 2 * Pi;
        currDir =  rotate(currDir, theta, phi);
        currPos = currPos + Normalize(currDir) * d;
        printf("Current direction: %.4f, %.4f, %.4f\n", 
                currDir.x, currDir.y, currDir.z);
        printf("Current position: %.4f, %.4f, %.4f\n", 
                currPos.x, currPos.y, currPos.z);

        haltonDim += 3;
    }

}

// Beam_SPPM Method Definitions
void Beam_SPPMIntegrator::Render(const Scene &scene) {
    ProfilePhase p(Prof::IntegratorRender);
    // Initialize _pixelBounds_ and _pixels_ array for Beam_SPPM
    Bounds2i pixelBounds = camera->film->croppedPixelBounds;
    int nPixels = pixelBounds.Area();
    std::unique_ptr<Beam_SPPMPixel[]> pixels(new Beam_SPPMPixel[nPixels]);
    for (int i = 0; i < nPixels; ++i) pixels[i].radius = initialSearchRadius;
    const Float invSqrtSPP = 1.f / std::sqrt(nIterations);
    pixelMemoryBytes = nPixels * sizeof(Beam_SPPMPixel);
    // Compute _lightDistr_ for sampling lights proportional to power
    std::unique_ptr<Distribution1D> lightDistr =
        ComputeLightPowerDistribution(scene);

    // Perform _nIterations_ of Beam_SPPM integration
    HaltonSampler sampler(nIterations, pixelBounds);

    // Compute number of tiles to use for Beam_SPPM camera pass
    Vector2i pixelExtent = pixelBounds.Diagonal();
    const int tileSize = 16;
    Point2i nTiles((pixelExtent.x + tileSize - 1) / tileSize,
                   (pixelExtent.y + tileSize - 1) / tileSize);
    ProgressReporter progress(2 * nIterations, "Rendering");
    for (int iter = 0; iter < nIterations; ++iter) {
        // Generate Beam_SPPM visible points
        std::vector<MemoryArena> perThreadArenas(MaxThreadIndex());
        {
            StatTimer timer(&hitPointTimer);
            ParallelFor2D([&](Point2i tile) {
                MemoryArena &arena = perThreadArenas[ThreadIndex];
                // Follow camera paths for _tile_ in image for Beam_SPPM
                int tileIndex = tile.y * nTiles.x + tile.x;
                std::unique_ptr<Sampler> tileSampler = sampler.Clone(tileIndex);

                // Compute _tileBounds_ for Beam_SPPM tile
                int x0 = pixelBounds.pMin.x + tile.x * tileSize;
                int x1 = std::min(x0 + tileSize, pixelBounds.pMax.x);
                int y0 = pixelBounds.pMin.y + tile.y * tileSize;
                int y1 = std::min(y0 + tileSize, pixelBounds.pMax.y);
                Bounds2i tileBounds(Point2i(x0, y0), Point2i(x1, y1));
                for (Point2i pPixel : tileBounds) {
                    // Prepare _tileSampler_ for _pPixel_
                    tileSampler->StartPixel(pPixel);
                    tileSampler->SetSampleNumber(iter);

                    // Generate camera ray for pixel for Beam_SPPM
                    CameraSample cameraSample =
                        tileSampler->GetCameraSample(pPixel);
                    RayDifferential ray;
                    Spectrum beta =
                        camera->GenerateRayDifferential(cameraSample, &ray);
                    ray.ScaleDifferentials(invSqrtSPP);

                    // Follow camera ray path until a visible point is created

                    // Get _Beam_SPPMPixel_ for _pPixel_
                    Point2i pPixelO = Point2i(pPixel - pixelBounds.pMin);
                    int pixelOffset =
                        pPixelO.x +
                        pPixelO.y * (pixelBounds.pMax.x - pixelBounds.pMin.x);
                    Beam_SPPMPixel &pixel = pixels[pixelOffset];
                    bool specularBounce = false;
                    for (int depth = 0; depth < maxDepth; ++depth) {
                        SurfaceInteraction isect;
                        ++totalPhotonSurfaceInteractions;
                        if (!scene.Intersect(ray, &isect)) {
                            // Accumulate light contributions for ray with no
                            // intersection
                            for (const auto &light : scene.lights)
                                pixel.Ld += beta * light->Le(ray);
                            break;
                        }
                        // Process Beam_SPPM camera ray intersection

                        // Compute BSDF at Beam_SPPM camera ray intersection
                        isect.ComputeScatteringFunctions(ray, arena, true);
                        if (!isect.bsdf) {
                            ray = isect.SpawnRay(ray.d);
                            --depth;
                            continue;
                        }
                        const BSDF &bsdf = *isect.bsdf;

                        // Accumulate direct illumination at Beam_SPPM camera ray
                        // intersection
                        Vector3f wo = -ray.d;
                        if (depth == 0 || specularBounce)
                            pixel.Ld += beta * isect.Le(wo); 
                        pixel.Ld += beta * UniformSampleOneLight(isect, 
                                        scene, arena, *tileSampler);
                        
                        // Possibly create visible point and end camera path
                        bool isDiffuse = bsdf.NumComponents(BxDFType(
                                             BSDF_DIFFUSE | BSDF_REFLECTION |
                                             BSDF_TRANSMISSION)) > 0;
                        bool isGlossy = bsdf.NumComponents(BxDFType(
                                            BSDF_GLOSSY | BSDF_REFLECTION |
                                            BSDF_TRANSMISSION)) > 0;
                        if (isDiffuse || (isGlossy && depth == maxDepth - 1)) {
                            pixel.vp = {isect.p, wo, beta, &bsdf};
                            break;
                        }
                        
                        // Spawn ray from Beam_SPPM camera path vertex
                        if (depth < maxDepth - 1) {
                            Float pdf;
                            Vector3f wi;
                            BxDFType type;
                            Spectrum f =
                                bsdf.Sample_f(wo, &wi, tileSampler->Get2D(),
                                              &pdf, BSDF_ALL, &type);
                            if (pdf == 0. || f.IsBlack()) break;
                            beta *= f * AbsDot(wi, isect.shading.n) / pdf;
                            specularBounce = (type & BSDF_SPECULAR) != 0;
                            // Account for subsurface scattering, if applicable
                            if (isect.bssrdf && (type & BSDF_TRANSMISSION)) {
                                pixel.vp = {isect.p, wo, beta, &bsdf, isect.bssrdf}; 
                                // Importance sample the BSSRDF
                                SurfaceInteraction pi;
                                Spectrum S = isect.bssrdf->Sample_S(
                                    scene, tileSampler->Get1D(), tileSampler->Get2D(), arena, &pi, &pdf);
                                printf("\nS spectrum: %.10f, %.10f, %.10f\npdf: %.10f\n", S[0], S[1], S[2], pdf);
                                DCHECK(!std::isinf(beta.y()));
                                if (S.IsBlack() || pdf == 0) break;
                                beta *= S / pdf;

                                // Account for the direct subsurface scattering component
                                pixel.Ld += beta * UniformSampleOneLight(pi, scene, arena, *tileSampler);
                                printf("Ld: %.10f, %.10f, %.10f\n", pixel.Ld[0], pixel.Ld[1], pixel.Ld[2]);
                                //break;
                            }

                            if (beta.y() < 0.25) {
                                Float continueProb =
                                    std::min((Float)1, beta.y());
                                if (tileSampler->Get1D() > continueProb) break;
                                beta /= continueProb;
                            }
                            ray = (RayDifferential)isect.SpawnRay(wi);

                        }
                    }
                }
            }, nTiles);
        }
        progress.Update();

        // Create grid of all Beam_SPPM visible points

        // Allocate grid for Beam_SPPM visible points
        int hashSize = nPixels;
        std::vector<std::atomic<Beam_SPPMPixelListNode *>> grid(hashSize);

        // Compute grid bounds for Beam_SPPM visible points
        Bounds3f gridBounds;
        Float maxRadius = 0.;
        for (int i = 0; i < nPixels; ++i) {
            const Beam_SPPMPixel &pixel = pixels[i];
            if (pixel.vp.beta.IsBlack()) continue;
            Bounds3f vpBound = Expand(Bounds3f(pixel.vp.p), pixel.radius);
            gridBounds = Union(gridBounds, vpBound);
            maxRadius = std::max(maxRadius, pixel.radius);
        }

        // Compute resolution of Beam_SPPM grid in each dimension
        Vector3f diag = gridBounds.Diagonal();
        Float maxDiag = MaxComponent(diag);
        int baseGridRes = (int)(maxDiag / maxRadius);
        //Assert(baseGridRes > 0);
        if (baseGridRes <= 0) return;
        int gridRes[3];
        for (int i = 0; i < 3; ++i)
            gridRes[i] = std::max((int)(baseGridRes * diag[i] / maxDiag), 1);

        // Add visible points to Beam_SPPM grid
        {
            StatTimer timer(&gridConstructionTimer);
            ParallelFor([&](int pixelIndex) {
                MemoryArena &arena = perThreadArenas[ThreadIndex];
                Beam_SPPMPixel &pixel = pixels[pixelIndex];
                if (!pixel.vp.beta.IsBlack()) {
                    // Add pixel's visible point to applicable grid cells
                    Float radius = pixel.radius;
                    Point3i pMin, pMax;
                    ToGrid(pixel.vp.p - Vector3f(radius, radius, radius),
                           gridBounds, gridRes, &pMin);
                    ToGrid(pixel.vp.p + Vector3f(radius, radius, radius),
                           gridBounds, gridRes, &pMax);
                    for (int z = pMin.z; z <= pMax.z; ++z)
                        for (int y = pMin.y; y <= pMax.y; ++y)
                            for (int x = pMin.x; x <= pMax.x; ++x) {
                                // Add visible point to grid cell $(x, y, z)$
                                int h = hash(Point3i(x, y, z), hashSize);
                                Beam_SPPMPixelListNode *node =
                                    arena.Alloc<Beam_SPPMPixelListNode>();
                                node->pixel = &pixel;

                                // Atomically add _node_ to the start of
                                // _grid[h]_'s linked list
                                node->next = grid[h];
                                while (grid[h].compare_exchange_weak(
                                           node->next, node) == false)
                                    ;
                            }
                    ReportValue(gridCellsPerVisiblePoint,
                                (1 + pMax.x - pMin.x) * (1 + pMax.y - pMin.y) *
                                    (1 + pMax.z - pMin.z));
                }
            }, nPixels, 4096);
        }

        // Trace photons and accumulate contributions
        {
            StatTimer timer(&photonTimer);
            std::vector<MemoryArena> photonShootArenas(MaxThreadIndex());
            ParallelFor([&](int photonIndex) {
                MemoryArena &arena = photonShootArenas[ThreadIndex];
                // Follow photon path for _photonIndex_
                uint64_t haltonIndex =
                    (uint64_t)iter * (uint64_t)photonsPerIteration +
                    photonIndex;
                int haltonDim = 0;

                // Choose light to shoot photon from
                Float lightPdf;
                Float lightSample = RadicalInverse(haltonDim++, haltonIndex);
                int lightNum =
                    lightDistr->SampleDiscrete(lightSample, &lightPdf);
                const std::shared_ptr<Light> &light = scene.lights[lightNum];

                // Compute sample values for photon ray leaving light source
                Point2f uLight0(RadicalInverse(haltonDim, haltonIndex),
                                RadicalInverse(haltonDim + 1, haltonIndex));
                Point2f uLight1(RadicalInverse(haltonDim + 2, haltonIndex),
                                RadicalInverse(haltonDim + 3, haltonIndex));
                Float uLightTime =
                    Lerp(RadicalInverse(haltonDim + 4, haltonIndex),
                         camera->shutterOpen, camera->shutterClose);
                haltonDim += 5;
                // Generate _photonRay_ from light source and initialize _beta_
                RayDifferential photonRay;
                Normal3f nLight;
                Float pdfPos, pdfDir;
                Spectrum Le =
                    light->Sample_Le(uLight0, uLight1, uLightTime, &photonRay,
                                     &nLight, &pdfPos, &pdfDir);
                printf("\tOrigin: %.3f, %.3f, %.3f\n", 
                        photonRay.o.x, photonRay.o.y, photonRay.o.z);
                printf("\tDirection: %.3f, %.3f, %.3f\n", 
                        photonRay.d.x, photonRay.d.y, photonRay.d.z);
                if (pdfPos == 0 || pdfDir == 0 || Le.IsBlack()) return;
                Spectrum beta = (AbsDot(nLight, photonRay.d) * Le) /
                                (lightPdf * pdfPos * pdfDir);
                if (beta.IsBlack()) return;

                // Follow photon path through scene and record intersections
                SurfaceInteraction isect;
                for (int depth = 0; depth < maxDepth; ++depth) {
                    if (!scene.Intersect(photonRay, &isect)) break;
                    ++totalPhotonSurfaceInteractions;
                    if (depth > 0) {//skip direct illumination
                        // Add photon contribution to nearby visible points
                        Point3i photonGridIndex;
                        if (ToGrid(isect.p, gridBounds, gridRes,
                                   &photonGridIndex)) {
                            int h = hash(photonGridIndex, hashSize);
                            // Add photon contribution to visible points in
                            // _grid[h]_
                            /*printf("Add photon contribution to visible points");
                            printf(" in _grid[%d]_\n", h);*/
                            for (Beam_SPPMPixelListNode *node =
                                     grid[h].load(std::memory_order_relaxed);
                                 node != nullptr; node = node->next) {
                                ++visiblePointsChecked;
                                /*printf("visiblePointsChecked: %d\n", 
                                        visiblePointsChecked);*/
                                Beam_SPPMPixel &pixel = *node->pixel;
                                Float radius = pixel.radius;
                                if (DistanceSquared(pixel.vp.p, isect.p) >
                                    radius * radius)
                                    continue;
                                // Update _pixel_ $\Phi$ and $M$ for nearby
                                // photon
                                Vector3f wi = -photonRay.d;
                                /*printf("Direction: %.3f, %.3f, %.3f\n", 
                                        wi.x, wi.y, wi.z);*/
                                Spectrum Phi;
                                /*printf("Is subsurface: %d\n", 
                                        pixel.vp.isSubsurface);*/
                                if ( !pixel.vp.isSubsurface ){
                                    Phi = beta * pixel.vp.bsdf->f(pixel.vp.wo, wi);
                                }else{
                                    if (pixel.vp.bssrdf != NULL){
                                        Phi = beta * pixel.vp.bssrdf->S(isect, wi);
                                    }
                                }
                                for (int i = 0; i < Spectrum::nSamples; ++i)
                                    pixel.Phi[i].Add(Phi[i]);
                                ++pixel.M;
                            }
                        }
                    }
                    // Sample new photon ray direction

                    // Compute BSDF at photon intersection point
                    isect.ComputeScatteringFunctions(photonRay, arena, true,
                                                     TransportMode::Importance);
                    if (!isect.bsdf) {
                        --depth;
                        photonRay = isect.SpawnRay(photonRay.d);
                        continue;
                    }
                    const BSDF &photonBSDF = *isect.bsdf;
                    const BSSRDF *photonBSSRDF = isect.bssrdf;

                    // Sample BSDF _fr_ and direction _wi_ for reflected photon
                    Vector3f wi, wo = -photonRay.d;
                    Float pdf;
                    BxDFType flags;

                    // Generate _bsdfSample_ for outgoing photon sample
                    Point2f bsdfSample(
                        RadicalInverse(haltonDim, haltonIndex),
                        RadicalInverse(haltonDim + 1, haltonIndex));
                    haltonDim += 2;
                    Spectrum fr = photonBSDF.Sample_f(wo, &wi, bsdfSample, &pdf,
                                                      BSDF_ALL, &flags);
                    if (fr.IsBlack() || pdf == 0.f) break;

                    if (photonBSSRDF != nullptr && (flags & BSDF_TRANSMISSION)){
                        bool absorbed = traceSubsurfaceScattering(isect, scene, 
                                                                  wo, 
                                                                  photonRay.o,
                                                                  &wi, 
                                                                  haltonDim, 
                                                                  haltonIndex,
                                                                  100);
                        //printf("Returned from trace photons\n");
                        if (absorbed) break;
                    }
                    Spectrum bnew =
                        beta * fr * AbsDot(wi, isect.shading.n) / pdf;

                    // Possibly terminate photon path with Russian roulette
                    Float q = std::max((Float)0, 1 - bnew.y() / beta.y());
                    if (RadicalInverse(haltonDim++, haltonIndex) < q) break;
                    beta = bnew / (1 - q);
                    photonRay = (RayDifferential)isect.SpawnRay(wi);
                }
                //printf("Exit depth loop\n");
                arena.Reset();
            }, photonsPerIteration, 8192);
            progress.Update();
            photonPaths += photonsPerIteration;
        }

        // Update pixel values from this pass's photons
        {
            StatTimer timer(&statsUpdateTimer);
            ParallelFor([&](int i) {
                Beam_SPPMPixel &p = pixels[i];
                if (p.M > 0) {
                    // Update pixel photon count, search radius, and $\tau$ from
                    // photons
                    Float gamma = (Float)2 / (Float)3;
                    Float Nnew = p.N + gamma * p.M;
                    Float Rnew = p.radius * std::sqrt(Nnew / (p.N + p.M));
                    Spectrum Phi;
                    for (int j = 0; j < Spectrum::nSamples; ++j)
                        Phi[j] = p.Phi[j];
                    p.tau = (p.tau + p.vp.beta * Phi) * (Rnew * Rnew) /
                            (p.radius * p.radius);
                    p.N = Nnew;
                    p.radius = Rnew;
                    p.M = 0;
                    for (int j = 0; j < Spectrum::nSamples; ++j)
                        p.Phi[j] = (Float)0;
                }
                // Reset _VisiblePoint_ in pixel
                p.vp.beta = 0.;
                p.vp.bsdf = nullptr;
            }, nPixels, 4096);
        }

        // Periodically store Beam_SPPM image in film and write image
        if (iter + 1 == nIterations || ((iter + 1) % writeFrequency) == 0) {
            int x0 = pixelBounds.pMin.x;
            int x1 = pixelBounds.pMax.x;
            uint64_t Np = (uint64_t)(iter + 1) * (uint64_t)photonsPerIteration;
            std::unique_ptr<Spectrum[]> image(new Spectrum[pixelBounds.Area()]);
            int offset = 0;
            for (int y = pixelBounds.pMin.y; y < pixelBounds.pMax.y; ++y) {
                for (int x = x0; x < x1; ++x) {
                    // Compute radiance _L_ for Beam_SPPM pixel _pixel_
                    const Beam_SPPMPixel &pixel =
                        pixels[(y - pixelBounds.pMin.y) * (x1 - x0) + (x - x0)];
                    Spectrum L = pixel.Ld / (iter + 1);
                    if (x > 9)
                        printf("%.3f ", L[0]);
                    L += pixel.tau / (Np * Pi * pixel.radius * pixel.radius);
                    image[offset++] = L;
                }
                printf("\n");
            }
            camera->film->SetImage(image.get());
            camera->film->WriteImage();
            // Write Beam_SPPM radius image, if requested
            if (getenv("Beam_SPPM_RADIUS")) {
                std::unique_ptr<Float[]> rimg(
                    new Float[3 * pixelBounds.Area()]);
                Float minrad = 1e30f, maxrad = 0;
                for (int y = pixelBounds.pMin.y; y < pixelBounds.pMax.y; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        const Beam_SPPMPixel &p =
                            pixels[(y - pixelBounds.pMin.y) * (x1 - x0) +
                                   (x - x0)];
                        minrad = std::min(minrad, p.radius);
                        maxrad = std::max(maxrad, p.radius);
                    }
                }
                fprintf(stderr,
                        "iterations: %d (%.2f s) radius range: %f - %f\n",
                        iter + 1, progress.ElapsedMS() / 1000., minrad, maxrad);
                int offset = 0;
                for (int y = pixelBounds.pMin.y; y < pixelBounds.pMax.y; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        const Beam_SPPMPixel &p =
                            pixels[(y - pixelBounds.pMin.y) * (x1 - x0) +
                                   (x - x0)];
                        Float v = 1.f - (p.radius - minrad) / (maxrad - minrad);
                        rimg[offset++] = v;
                        rimg[offset++] = v;
                        rimg[offset++] = v;
                    }
                }
                Point2i res(pixelBounds.pMax.x - pixelBounds.pMin.x,
                            pixelBounds.pMax.y - pixelBounds.pMin.y);
                WriteImage("sppm_radius.png", rimg.get(), pixelBounds, res);
            }
        }
    }
    progress.Done();
}

Integrator *CreateBeam_SPPMIntegrator(const ParamSet &params,
                                 std::shared_ptr<const Camera> camera) {
    int nIterations = params.FindOneInt("numiterations", 64);
    int maxDepth = params.FindOneInt("maxdepth", 5);
    int photonsPerIter = params.FindOneInt("photonsperiteration", -1);
    int writeFreq = params.FindOneInt("imagewritefrequency", 1 << 31);
    Float radius = params.FindOneFloat("radius", 1.f);
    if (PbrtOptions.quickRender) nIterations = std::max(1, nIterations / 16);
    return new Beam_SPPMIntegrator(camera, nIterations, photonsPerIter, maxDepth,
                              radius, writeFreq);
}
