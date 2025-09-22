#pragma once

#include <GL/glew.h>
#include <freetype/freetype.h>

#include <string>
#include <vector>
#include <ranges>

#include "Geometry.h"

class Text {
public:
	
	QuadManager::QuadReference mQuadReference;

	GLuint mTexture = 0;
	FT_Int mTotalWidth = 0, mTotalHeight = 0, mMaxAscent = 0, mMaxDescent = 0;
	std::vector<std::uint32_t> mTextBuffer;

	void create( const std::string& text, const FT_Face& face) {

		mTotalWidth = 0;
		mMaxAscent = 0;
		mMaxDescent = 0;
		
		for (auto c: text ){

			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;

			mTotalWidth += g->advance.x >> 6; // advance in pixels
			mMaxAscent = std::max(mMaxAscent, g->bitmap_top);
			mMaxDescent = std::max(mMaxDescent, FT_Int(g->bitmap.rows - g->bitmap_top));
		}
		mTotalHeight = mMaxAscent + mMaxDescent;

		std::uint32_t greyColor = 0xFFFFFFFF;//opaque with white background with grey forground/text
		std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(&greyColor);

		//add spaces between glyphs and padding of line and bottom of line
		mTotalWidth += text.size();
		mTotalHeight += 2;
		mTextBuffer.resize(mTotalWidth * mTotalHeight, greyColor);

		int penX = 0;
		for (std::size_t i : std::views::iota(0ULL, text.size())) {

			char c = text[i];

			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;
			FT_Bitmap& bmp = g->bitmap;

			int xOffset = penX + g->bitmap_left;
			int yOffset = mMaxAscent - g->bitmap_top;

			for (int row = 0; row < bmp.rows; ++row) {
				for (int col = 0; col < bmp.width; ++col) {
					int x = 1 + xOffset + col + i; //+i/1 == space between glyphs
					int y = 1 + yOffset + row;

					float greyScale = 1.0f - bmp.buffer[row * bmp.pitch + col] / 255.0f;
					std::uint8_t* savedColor = reinterpret_cast<std::uint8_t*>(&mTextBuffer[y * mTotalWidth + x]);

					savedColor[0] = bytes[0] * greyScale; // R
					savedColor[1] = bytes[1] * greyScale;   // G
					savedColor[2] = bytes[2] * greyScale; // B
					savedColor[3] = bytes[3]; // A
				}
			}

			penX += g->advance.x >> 6;
		}
	
		if (mTexture)
			glDeleteTextures(1, &mTexture);

		glGenTextures(1, &mTexture);

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, mTexture);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mTotalWidth, mTotalHeight, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, mTextBuffer.data());

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	void destroy(){
		if (mTexture) {
			glDeleteTextures(1, &mTexture);
			mTexture = 0;
		}
		mTextBuffer.clear();
		mTextBuffer.shrink_to_fit();
	}
	
	float getAspectRatio() {
		return mTotalWidth / float(mTotalHeight);
	}
};

class TextManager {

public:

	QuadManager* mQuadManager;

	FT_Library mFreeType;
	FT_Face mFontFace;

	std::vector<Text> mStaticText;
	float mScale = 0.01f;

	float mInsertY = -1.0f; //start at bottom

	void create(QuadManager* quadManager, const std::string& fontName, FT_UInt fontSize, float scale = 0.01f) {

		mQuadManager = quadManager;
		mScale = scale;

		FT_Init_FreeType(&mFreeType);

		FT_New_Face(mFreeType, fontName.c_str(), 0, &mFontFace);
		FT_Set_Pixel_Sizes(mFontFace, 0, fontSize);
	}

	void destroy() {

		FT_Done_Face(mFontFace);       // Destroys the font face
		FT_Done_FreeType(mFreeType);     // Shuts down the FreeType library
	}

	void addStaticText(const std::string& text) {
		Text t;
		t.create(text, mFontFace);
		t.mQuadReference = mQuadManager->add(-1.0, mInsertY, t.getAspectRatio(), mScale);

		auto yHalfHeight = mScale;
		mInsertY += 2.0f * yHalfHeight;

		mStaticText.push_back(std::move(t));
	}

	void render() {

		auto& qm = *mQuadManager;

		for (auto& text : mStaticText) {

			//glEnable(GL_BLEND);
			//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

			glBindTexture(GL_TEXTURE_2D, text.mTexture);

			qm.render(text.mQuadReference);
		}
	} 
};