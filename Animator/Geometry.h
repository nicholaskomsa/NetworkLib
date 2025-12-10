#pragma once

#include <GL/glew.h>
#include <vector>
#include <array>

class QuadManager {
public:

	using Vertices = std::vector<float>;
	Vertices mVertices;
    GLuint mVao = 0, mVbo = 0;

    static constexpr auto QuadSize = 16;
    using Quad = std::array<float, QuadSize>;

    using QuadReference = std::size_t;
    QuadReference mNewQuadIdx = 0;


    void reserve(std::size_t quadCount = 1) {
        mVertices.reserve(quadCount * sizeof(Quad));                     
        mNewQuadIdx = 0;
        mVertices.clear();

    }

    QuadReference addQuad(const Quad& quad) {

        mVertices.insert(mVertices.end(), quad.begin(), quad.end());

        return mNewQuadIdx++;
    }

    void setAspectRatio(QuadReference quadRef, float monitorRatio, float desiredRatio) {
       
		float* quad = mVertices.data() + quadRef * QuadSize;
      
        auto yScale = 1.0f;
        auto xScale = desiredRatio / monitorRatio;
    
        quad[0] = -xScale; quad[1] = yScale;
        quad[4] = xScale; quad[5] = yScale;
        quad[8] = -xScale; quad[9] = -yScale;
        quad[12] = xScale; quad[13] = -yScale;
    }

    QuadReference addIdentity() {

        Quad quad = {
            //vertex = X, Y, U, V
            -1.0f, 1.0f,    0.0f, 0.0f  // Top-left
            , 1.0f, 1.0f,   1.0f, 0.0f  // Top-right
            , -1.0f, -1.0f, 0.0f, 1.0f  // Bottom-left
            , 1.0f, -1.0f,  1.0f, 1.0f };   // Bottom-right

        return addQuad(quad);
    }

    QuadReference add(float x, float y, float aspectRatio, float scale = 1.0f) {

        auto yHalfHeight = scale;
        auto centerToTopLeftY = y + yHalfHeight;

        auto xHalfWidth = aspectRatio * yHalfHeight;
        auto centerToTopLeftX = x + xHalfWidth;

        Quad quad = { //this quad is a triangle strip
            //vertex = X, Y, U, V
              centerToTopLeftX - xHalfWidth,  centerToTopLeftY + yHalfHeight,   0.0f, 0.0f  // Top-left
            , centerToTopLeftX + xHalfWidth,  centerToTopLeftY + yHalfHeight,   1.0f, 0.0f  // Top-right
            , centerToTopLeftX - xHalfWidth,  centerToTopLeftY - yHalfHeight,   0.0f, 1.0f  // Bottom-left
            , centerToTopLeftX + xHalfWidth,  centerToTopLeftY - yHalfHeight,   1.0f, 1.0f };   // Bottom-right

        return addQuad(quad);
    }

    void generate() {

        glGenVertexArrays(1, &mVao);
        glGenBuffers(1, &mVbo);

        glBindVertexArray(mVao);
        glBindBuffer(GL_ARRAY_BUFFER, mVbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * mVertices.size(), mVertices.data(), GL_STATIC_DRAW);

        // Position Attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
        glEnableVertexAttribArray(0);

        // Texture Coordinate Attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    void destroy() {

        if (mVbo) {
            glDeleteBuffers(1, &mVbo);
            mVbo = 0;
        }
        if (mVao) {
            glDeleteVertexArrays(1, &mVao);
            mVao = 0;
        }
        mVertices.clear();
        mVertices.shrink_to_fit();
    }

    void render(QuadReference quad) {
        glDrawArrays(GL_TRIANGLE_STRIP, quad * 4, 4);
    }
};